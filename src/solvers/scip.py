import functools
import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pyscipopt import (
    SCIP_EVENTTYPE,
    SCIP_PARAMSETTING,
    SCIP_RESULT,
    Branchrule,
    Eventhdlr,
    Model,
    quicksum,
)
from pyscipopt.scip import Event, Variable

from src.solvers.lp_utils import solve_lp_with_conds, var_name_to_id
from src.solvers.solver_interface import (
    Bounds,
    BranchCond,
    Candidate,
    LPNode,
    Solver,
)

type PoolMode = Literal["leaves", "mixed"]

_INVALID_OBJ = -1e18  # threshold below which an LP objective is considered garbage
_POOL_MODES: tuple[PoolMode, ...] = ("leaves", "mixed")
_MIN_MIXED_POOL_SIZE = 4

logger = logging.getLogger(__name__)


class SCIPCallbackError(RuntimeError):
    """An exception raised inside a SCIP callback (event handler or branching rule)."""


def _capture_callback_errors(
    fallback: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """PySCIPOpt silently swallows exceptions raised inside callbacks, so record the
    first one on the plugin; SCIPSolver re-raises it once optimize() returns."""

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised after optimize()
                if self.callback_error is None:
                    self.callback_error = exc
                return fallback

        return wrapper

    return decorator


def _raise_callback_errors(
    *plugins: NodeTracker | BNBMostFractionalBranchrule | None,
) -> None:
    for plugin in plugins:
        if plugin is not None and plugin.callback_error is not None:
            raise SCIPCallbackError(
                f"{type(plugin).__name__} failed during the SCIP solve"
            ) from plugin.callback_error


class NodeTracker(Eventhdlr):
    """Records every node SCIP visits: its branch path, local bounds and, once its
    LP has been solved (LPSOLVED event), the LP primal/dual solution."""

    def __init__(
        self,
        integer: list[int],
        ineq_cons_names: list[str] | None = None,
        eq_cons_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.integer = integer
        self.ineq_cons_names = list(ineq_cons_names or [])
        self.eq_cons_names = list(eq_cons_names or [])
        self.node_map = {}
        self.parent_bounds = {}
        self.event_counts = {
            "NODEFOCUSED": 0,
            "NODESOLVED": 0,
            "NODEDELETE": 0,
            "LPSOLVED": 0,
        }
        self.skipped_invalid_lp_obj = 0
        self.lp_obj_ok = 0
        self.callback_error = None

    @_capture_callback_errors()
    def eventinit(self) -> None:
        for event_type in (
            SCIP_EVENTTYPE.NODEFOCUSED,
            SCIP_EVENTTYPE.NODESOLVED,
            SCIP_EVENTTYPE.NODEDELETE,
            SCIP_EVENTTYPE.LPSOLVED,
        ):
            self.model.catchEvent(event_type, self)

    def eventinitsol(self) -> None:
        pass

    def eventexitsol(self) -> None:
        pass

    @_capture_callback_errors()
    def eventexec(self, event: Event) -> None:
        model = self.model
        event_type = event.getType()

        if event_type == SCIP_EVENTTYPE.NODEFOCUSED:
            self.event_counts["NODEFOCUSED"] += 1
            self._handle_node_event(model, event, status="focused")
            return

        if event_type == SCIP_EVENTTYPE.NODESOLVED:
            self.event_counts["NODESOLVED"] += 1
            self._handle_node_event(model, event, status="solved")
            return

        if event_type == SCIP_EVENTTYPE.NODEDELETE:
            self.event_counts["NODEDELETE"] += 1
            self._handle_delete_event(event)
            return

        if event_type == SCIP_EVENTTYPE.LPSOLVED:
            self.event_counts["LPSOLVED"] += 1
            self._handle_node_event(model, event, status="lp_solved")

    def _handle_node_event(self, model: Model, event: Event, status: str) -> None:
        # LPSOLVED events carry no node, so fall back to the node being processed.
        node = event.getNode()
        if node is None:
            node = model.getCurrentNode()
        if node is None:
            return

        node_num = node.getNumber()
        depth = node.getDepth()
        parent = node.getParent()
        parent_num = parent.getNumber() if parent is not None else None

        entry = {
            "node_num": node_num,
            "depth": depth,
            "parent": parent_num,
            "lp_obj": None,
            "status": status,
            "was_solved": status == "solved",
            "was_lp_solved": status == "lp_solved",
            "was_deleted": False,
            "branching_decision": None,
            "node_bounds": {},
            "conds": [],
            "primal": {},
            "duals_ineq": {},
            "duals_eq": {},
            "reduced_costs": {},
        }

        lp_obj = float(model.getLPObjVal())
        if lp_obj >= _INVALID_OBJ:
            self.lp_obj_ok += 1
            entry["lp_obj"] = lp_obj
        else:
            self.skipped_invalid_lp_obj += 1

        local_conds = []
        branchings = node.getParentBranchings()
        if branchings is not None:
            vars_b, bounds_b, bound_types = branchings
            for var_obj, bound_val, bound_type in zip(vars_b, bounds_b, bound_types):
                var_id = var_name_to_id(var_obj.name)
                op = ">=" if int(bound_type) == 0 else "<="
                local_conds.append((var_id, op, float(bound_val)))
            if local_conds:
                last = local_conds[-1]
                entry["branching_decision"] = {
                    "var": last[0],
                    "type": last[1],
                    "value": last[2],
                }

        node_bounds = {
            var.name: {"lb": var.getLbLocal(), "ub": var.getUbLocal()}
            for var in model.getVars()
        }
        entry["node_bounds"] = node_bounds

        if (
            node_bounds
            and not local_conds
            and parent_num is not None
            and parent_num in self.parent_bounds
        ):
            for var_name, bound_info in node_bounds.items():
                parent_lb, parent_ub = self.parent_bounds[parent_num].get(
                    var_name, (None, None)
                )
                curr_lb, curr_ub = bound_info.get("lb"), bound_info.get("ub")
                var_id = var_name_to_id(var_name)
                if (
                    curr_lb is not None
                    and parent_lb is not None
                    and curr_lb > parent_lb + 1e-6
                ):
                    entry["branching_decision"] = {
                        "var": var_id,
                        "type": ">=",
                        "value": curr_lb,
                    }
                    local_conds = [(var_id, ">=", curr_lb)]
                    break
                if (
                    curr_ub is not None
                    and parent_ub is not None
                    and curr_ub < parent_ub - 1e-6
                ):
                    entry["branching_decision"] = {
                        "var": var_id,
                        "type": "<=",
                        "value": curr_ub,
                    }
                    local_conds = [(var_id, "<=", curr_ub)]
                    break

        conds = (
            list(self.node_map[parent_num].get("conds", []))
            if parent_num in self.node_map
            else []
        )
        conds.extend(local_conds)
        entry["conds"] = conds

        self.parent_bounds[node_num] = {
            var_name: (bound_info.get("lb"), bound_info.get("ub"))
            for var_name, bound_info in node_bounds.items()
        }

        if status == "lp_solved":
            for var in model.getVars():
                entry["primal"][var.name] = float(var.getLPSol())

            for cons in model.getConss():
                name = cons.name
                if name in self.eq_cons_names:
                    entry["duals_eq"][name] = float(model.getDualsolLinear(cons))
                elif name in self.ineq_cons_names:
                    entry["duals_ineq"][name] = float(model.getDualsolLinear(cons))

            for var in model.getVars():
                entry["reduced_costs"][var.name] = float(model.getVarRedcost(var))

        existing = self.node_map.get(node_num, {})
        if existing:
            if entry["lp_obj"] is None and existing.get("lp_obj") is not None:
                entry["lp_obj"] = existing["lp_obj"]

            entry["was_solved"] = bool(
                existing.get("was_solved", False) or entry.get("was_solved", False)
            )
            entry["was_lp_solved"] = bool(
                existing.get("was_lp_solved", False)
                or entry.get("was_lp_solved", False)
            )
            entry["was_deleted"] = bool(
                existing.get("was_deleted", False) or entry.get("was_deleted", False)
            )

            for key in (
                "primal",
                "duals_ineq",
                "duals_eq",
                "reduced_costs",
                "node_bounds",
            ):
                current = entry.get(key, {}) or {}
                previous = existing.get(key, {}) or {}
                if previous:
                    merged = dict(previous)
                    merged.update({k: v for k, v in current.items() if v is not None})
                    entry[key] = merged

            if not entry.get("conds") and existing.get("conds"):
                entry["conds"] = list(existing["conds"])
            if (
                entry.get("branching_decision") is None
                and existing.get("branching_decision") is not None
            ):
                entry["branching_decision"] = existing["branching_decision"]
        self.node_map[node_num] = entry

    def _handle_delete_event(self, event: Event) -> None:
        node = event.getNode()
        if node is None:
            return
        entry = self.node_map.get(node.getNumber())
        if entry is None:
            return
        entry["was_deleted"] = True


class BNBMostFractionalBranchrule(Branchrule):
    def __init__(self) -> None:
        super().__init__()
        self.callback_error = None

    @staticmethod
    def _bnb_fractionality(value: float) -> float:
        abs_value = abs(float(value))
        return abs_value - np.floor(abs_value)

    @_capture_callback_errors(fallback={"result": SCIP_RESULT.DIDNOTRUN})
    def branchexeclp(self, allowaddcons: bool) -> dict[str, int]:
        branch_cands, branch_cand_sols, _, _, npriocands, _ = (
            self.model.getLPBranchCands()
        )

        if npriocands <= 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = None
        best_key = None
        for idx in range(npriocands):
            sol_value = float(branch_cand_sols[idx])
            frac = self._bnb_fractionality(sol_value)
            if frac < 1e-6:
                continue

            var_id = var_name_to_id(branch_cands[idx].name)
            tie_break = int(var_id) if isinstance(var_id, int) else int(1e9)
            key = (frac, -tie_break)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self.model.branchVarVal(
            branch_cands[best_idx], float(branch_cand_sols[best_idx])
        )
        return {"result": SCIP_RESULT.BRANCHED}

    def branchexecext(self, allowaddcons: bool) -> dict[str, int]:
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def branchexecps(self, allowaddcons: bool) -> dict[str, int]:
        return {"result": SCIP_RESULT.DIDNOTRUN}


class _CandidatePool:
    """Ordered pool of solver candidates, deduplicated on x."""

    def __init__(
        self,
        debug: dict[str, int],
        check: Callable[[Candidate], bool] | None = None,
    ) -> None:
        self.results = []
        self.debug = debug
        self._check = check
        self._seen_x = set()

    def __len__(self) -> int:
        return len(self.results)

    def add(self, entry: Candidate | None, source: str) -> bool:
        if entry is None or entry.get("x") is None:
            return False
        if self._check is not None and not self._check(entry):
            self.debug["rejected_candidates"] += 1
            return False
        signature = tuple(np.round(np.asarray(entry["x"], dtype=float), 8).tolist())
        if signature in self._seen_x:
            self.debug["duplicate_candidates"] += 1
            return False
        self._seen_x.add(signature)
        self.results.append(entry)
        key = f"added_{source}"
        self.debug[key] = self.debug.get(key, 0) + 1
        return True


def _new_pool_debug() -> dict[str, int]:
    return {
        "duplicate_candidates": 0,
        "rejected_candidates": 0,
        "pre_filter_results": 0,
        "post_filter_results": 0,
        "filter_removed": 0,
        "filter_fallback_to_ordered": 0,
    }


class SCIPSolver(Solver):
    """MILP solver that runs SCIP's branch and bound and turns the explored tree into
    a pool of KKT-valid LP candidates.

    pool_mode selects how the pool is built:
      - "leaves": the leaf nodes of the explored tree, each re-solved as an LP with its
        branch conditions. All entries are marked fathomed.
      - "mixed": the root LP relaxation, SCIP's best integer solution, an LP for every
        branch path, the LP solution SCIP found at every node, and pseudo-branch LPs
        around the best solution if the pool has fewer than 4 entries.
    """

    def __init__(
        self,
        verbose: bool = False,
        pool_mode: PoolMode = "leaves",
        disable_heuristics: bool = True,
        disable_presolve: bool = True,
        disable_separating: bool = True,
        disable_propagation: bool = False,
        disable_conflict_analysis: bool = False,
        disable_symmetry: bool = False,
        prefer_most_fractional_branching: bool = False,
        prefer_breadth_first: bool = False,
        tighten_integer_projected_bounds: bool = False,
        mimic_bnb_pool_filter: bool = False,
        prefer_depth_first: bool = True,
    ) -> None:
        if pool_mode not in _POOL_MODES:
            raise ValueError(
                f"pool_mode must be one of {_POOL_MODES}. Got '{pool_mode}'."
            )
        self.verbose = verbose
        self.pool_mode = pool_mode
        self.disable_heuristics = disable_heuristics
        self.disable_presolve = disable_presolve
        self.disable_separating = disable_separating
        self.disable_propagation = disable_propagation
        self.disable_conflict_analysis = disable_conflict_analysis
        self.disable_symmetry = disable_symmetry
        self.prefer_most_fractional_branching = prefer_most_fractional_branching
        self.prefer_breadth_first = prefer_breadth_first
        self.tighten_integer_projected_bounds = tighten_integer_projected_bounds
        self.mimic_bnb_pool_filter = mimic_bnb_pool_filter
        self.prefer_depth_first = prefer_depth_first
        self.last_tree_snapshot = None

    @staticmethod
    def _format_conds(conds: list[BranchCond]) -> str:
        if not conds:
            return "root"
        return ", ".join(
            f"x_{var_name_to_id(var_id)} {op} {float(val):.6g}"
            for var_id, op, val in conds
        )

    def _store_tree_snapshot(
        self,
        tracker: NodeTracker,
        results: list[Candidate],
        solve_status: str,
        pool_debug: dict[str, int] | None = None,
    ) -> None:
        pool_node_labels = []
        pool_node_label_set = set()

        def _append_pool_label(label: int | str) -> None:
            label = str(label)
            if label in pool_node_label_set:
                return
            pool_node_label_set.add(label)
            pool_node_labels.append(label)

        nodes = []
        for node_num, info in sorted(
            tracker.node_map.items(),
            key=lambda item: (
                item[1].get("depth") if item[1].get("depth") is not None else -1,
                item[0],
            ),
        ):
            nodes.append(
                {
                    "node_num": int(node_num),
                    "parent": info.get("parent"),
                    "depth": info.get("depth"),
                    "lp_obj": info.get("lp_obj"),
                    "status": info.get("status"),
                    "was_solved": bool(info.get("was_solved", False)),
                    "was_lp_solved": bool(info.get("was_lp_solved", False)),
                    "was_deleted": bool(info.get("was_deleted", False)),
                    "branching_decision": info.get("branching_decision"),
                    "conds": list(info.get("conds", [])),
                    "in_pool": False,
                }
            )

        node_lookup = {str(node["node_num"]): node for node in nodes}
        extra_counter = 0
        for entry in results or []:
            node_num = entry.get("node_num")
            if node_num is not None:
                node_label = str(int(node_num))
                _append_pool_label(node_label)
                if node_label in node_lookup:
                    node_lookup[node_label]["in_pool"] = True
                continue

            extra_counter += 1
            node_label = f"candidate_{extra_counter}"
            _append_pool_label(node_label)
            conds = list(entry.get("conds", []))
            branch = conds[-1] if conds else None
            nodes.append(
                {
                    "node_num": node_label,
                    "parent": "root" if len(conds) <= 1 else f"depth_{len(conds) - 1}",
                    "depth": len(conds),
                    "lp_obj": entry.get("fun"),
                    "status": entry.get("status"),
                    "branching_decision": None
                    if branch is None
                    else {"var": branch[0], "type": branch[1], "value": branch[2]},
                    "conds": conds,
                    "in_pool": True,
                }
            )

        nodes.sort(
            key=lambda node: (
                node.get("depth") if node.get("depth") is not None else -1,
                str(node.get("node_num")),
            )
        )

        self.last_tree_snapshot = {
            "solve_status": solve_status,
            "pool_node_labels": pool_node_labels,
            "nodes": nodes,
            "event_counts": dict(tracker.event_counts),
            "pool_debug": dict(pool_debug or {}),
        }

    def format_last_tree_log(self) -> str:
        snapshot = self.last_tree_snapshot
        if not snapshot:
            return "[scip-tree] unavailable"

        lines = [
            (
                "[scip-tree] "
                f"status={snapshot['solve_status']} "
                f"tracked_nodes={len(snapshot['nodes'])} "
                f"pool_nodes={len(snapshot['pool_node_labels'])} "
                f"focused={snapshot['event_counts'].get('NODEFOCUSED', 0)} "
                f"solved={snapshot['event_counts'].get('NODESOLVED', 0)} "
                f"deleted={snapshot['event_counts'].get('NODEDELETE', 0)}"
            )
        ]

        if snapshot["pool_node_labels"]:
            lines.append(
                "[scip-tree-pool] node_ids=" + ", ".join(snapshot["pool_node_labels"])
            )
        else:
            lines.append("[scip-tree-pool] node_ids=none")

        pool_debug = snapshot.get("pool_debug") or {}
        if pool_debug:
            lines.append(
                "[scip-tree-pool-debug] "
                + " ".join(
                    f"{key}={value}" for key, value in sorted(pool_debug.items())
                )
            )

        for node in snapshot["nodes"]:
            lp_obj = node.get("lp_obj")
            lp_obj_str = "None" if lp_obj is None else f"{float(lp_obj):.6f}"
            branch = node.get("branching_decision")
            if branch is None:
                branch_str = "root"
            else:
                branch_str = f"x_{var_name_to_id(branch['var'])} {branch['type']} {float(branch['value']):.6g}"
            lines.append(
                "[scip-tree-node] "
                f"node={node['node_num']} "
                f"parent={node.get('parent')} "
                f"depth={node.get('depth')} "
                f"status={node.get('status')} "
                f"was_solved={node.get('was_solved')} "
                f"was_lp_solved={node.get('was_lp_solved')} "
                f"was_deleted={node.get('was_deleted')} "
                f"in_pool={node.get('in_pool')} "
                f"lp_obj={lp_obj_str} "
                f"branch={branch_str} "
                f"conds={self._format_conds(node.get('conds', []))}"
            )

        return "\n".join(lines)

    @staticmethod
    def _cleanup_model(model: Model) -> None:
        # Runs in a finally block, so a failure here must not mask an exception that
        # is already propagating; log it instead.
        try:
            # After a normal solve, free the transformed problem first, then the
            # original problem once SCIP is back in PROBLEM stage.
            if model.getStageName() == "SOLVED":
                model.freeTransform()
            if model.getStageName() == "PROBLEM":
                model.freeProb()
        except Exception:
            logger.warning("Failed to free the SCIP model.", exc_info=True)

    def _configure_model(self, model: Model) -> BNBMostFractionalBranchrule | None:
        """Apply the solver settings. Returns the custom branching rule, if any.

        setParam raises KeyError for an unknown parameter name, so a typo fails
        loudly instead of silently leaving SCIP's default in place.
        """
        if not self.verbose:
            model.hideOutput()

        if self.disable_presolve:
            model.setPresolve(SCIP_PARAMSETTING.OFF)

        if self.disable_separating:
            model.setSeparating(SCIP_PARAMSETTING.OFF)

        if self.disable_heuristics:
            model.setHeuristics(SCIP_PARAMSETTING.OFF)

        if self.disable_propagation:
            model.disablePropagation()

        if self.disable_conflict_analysis:
            model.setParam("conflict/enable", False)

        if self.disable_symmetry:
            model.setParam("misc/usesymmetry", 0)

        branchrule = None
        if self.prefer_most_fractional_branching:
            branchrule = BNBMostFractionalBranchrule()
            model.includeBranchrule(
                branchrule,
                "bnb_most_fractional",
                "branch like bnb.py on the variable with largest fractional part",
                priority=10_000_000,
                maxdepth=-1,
                maxbounddist=1,
            )
            model.setParam("branching/leastinf/priority", 1000000)
            model.setParam("branching/pscost/priority", -1000000)
            model.setParam("branching/relpscost/priority", -1000000)
            model.setParam("branching/inference/priority", -1000000)

        if self.prefer_breadth_first:
            model.setParam("nodeselection/bfs/stdpriority", 1000000)
            model.setParam("nodeselection/dfs/stdpriority", -1000000)
            model.setParam("nodeselection/estimate/stdpriority", -1000000)
            model.setParam("nodeselection/hybridestim/stdpriority", -1000000)
        elif self.prefer_depth_first:
            model.setParam("nodeselection/dfs/stdpriority", 1000000)

        return branchrule

    @staticmethod
    def _build_model(
        model: Model, init_node: LPNode
    ) -> tuple[list[Variable], list[str], list[str]]:
        c = init_node["c"]
        a_ub = init_node["A_ub"]
        b_ub = init_node["b_ub"]
        a_eq = init_node["A_eq"]
        b_eq = init_node["b_eq"]
        bounds = init_node["bounds"]
        integer = init_node["integer"]
        n_vars = len(c)

        vars_ = []
        for idx in range(n_vars):
            lb, ub = bounds[idx]
            vars_.append(
                model.addVar(
                    name=f"x_{idx}",
                    lb=float(lb) if lb is not None else -1e20,
                    ub=float(ub) if ub is not None else 1e20,
                    vtype="I" if integer[idx] else "C",
                )
            )

        model.setObjective(
            quicksum(float(c[idx]) * vars_[idx] for idx in range(n_vars)),
            sense="minimize",
        )

        ineq_cons_names = []
        if a_ub is not None and b_ub is not None:
            for row_idx in range(a_ub.shape[0]):
                expr = quicksum(
                    float(a_ub[row_idx, col_idx]) * vars_[col_idx]
                    for col_idx in range(n_vars)
                    if a_ub[row_idx, col_idx] != 0
                )
                cons = model.addCons(expr <= float(b_ub[row_idx]))
                ineq_cons_names.append(cons.name)

        eq_cons_names = []
        if a_eq is not None and b_eq is not None:
            for row_idx in range(a_eq.shape[0]):
                expr = quicksum(
                    float(a_eq[row_idx, col_idx]) * vars_[col_idx]
                    for col_idx in range(n_vars)
                    if a_eq[row_idx, col_idx] != 0
                )
                cons = model.addCons(expr == float(b_eq[row_idx]))
                eq_cons_names.append(cons.name)

        return vars_, ineq_cons_names, eq_cons_names

    @staticmethod
    def _project_integer_bounds(
        x: np.ndarray, bounds: Bounds, integer: list[int]
    ) -> Bounds:
        projected_bounds = list(bounds)
        x = np.asarray(x, dtype=float)

        for idx, is_integer in enumerate(integer):
            if not is_integer or idx >= len(projected_bounds):
                continue

            lb, ub = projected_bounds[idx]
            projected_value = float(np.round(x[idx]))
            if lb is not None:
                projected_value = max(float(lb), projected_value)
            if ub is not None:
                projected_value = min(float(ub), projected_value)
            projected_bounds[idx] = (projected_value, projected_value)

        return projected_bounds

    @staticmethod
    def _all_integer(x: np.ndarray, integer: list[int], tol: float = 1e-6) -> bool:
        x = np.asarray(x, dtype=float)
        return all(
            (abs(float(value) - round(float(value))) < tol) if is_integer else True
            for value, is_integer in zip(x, integer)
        )

    @staticmethod
    def _bnb_like_order_key(entry: Candidate) -> tuple[int, int]:
        depth = entry.get("depth")
        node_num = entry.get("node_num")
        return (
            int(depth) if depth is not None else -1,
            int(node_num) if node_num is not None else 10**9,
        )

    def _filter_pool_like_bnb(
        self, results: list[Candidate], integer: list[int]
    ) -> tuple[list[Candidate], dict[str, int]]:
        ordered_results = sorted(results, key=self._bnb_like_order_key)
        filtered = []
        incumbent = float("inf")
        stats = {
            "ordered_results": len(ordered_results),
            "integral_kept": 0,
            "integral_skipped_nonimproving": 0,
            "fractional_kept_fathomed": 0,
            "fractional_skipped_open": 0,
            "fallback_to_ordered": 0,
        }

        for entry in ordered_results:
            fun_value = float(entry["fun"])
            is_integer = self._all_integer(entry["x"], integer)

            if is_integer:
                if fun_value <= incumbent:
                    incumbent = fun_value
                    kept = dict(entry)
                    kept["fathomed"] = False
                    filtered.append(kept)
                    stats["integral_kept"] += 1
                else:
                    stats["integral_skipped_nonimproving"] += 1
                continue

            if incumbent < float("inf") and fun_value > incumbent:
                kept = dict(entry)
                kept["fathomed"] = True
                filtered.append(kept)
                stats["fractional_kept_fathomed"] += 1
            else:
                stats["fractional_skipped_open"] += 1

        if filtered:
            stats["returned_results"] = len(filtered)
            return filtered, stats

        stats["fallback_to_ordered"] = 1
        stats["returned_results"] = len(ordered_results)
        return ordered_results, stats

    @staticmethod
    def _is_valid_native_candidate(x: np.ndarray, fun_value: float) -> bool:
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            return False
        if np.any(np.abs(x) >= 1e19):
            return False
        if not np.isfinite(fun_value):
            return False
        return not fun_value <= _INVALID_OBJ

    @staticmethod
    def _leaf_node_ids(tracker: NodeTracker) -> set[int]:
        child_counts = {}
        for node_num in tracker.node_map:
            child_counts[int(node_num)] = 0

        for node_num, info in tracker.node_map.items():
            parent = info.get("parent")
            if parent is None:
                continue
            parent_num = int(parent)
            child_counts[parent_num] = child_counts.get(parent_num, 0) + 1

        return {int(node_num) for node_num, count in child_counts.items() if count == 0}

    @staticmethod
    def _pseudo_branch_conds(
        init_node: LPNode, x_ref: np.ndarray, max_vars: int = 6
    ) -> list[list[BranchCond]]:
        """Generate single-variable branch constraints from a reference solution.

        This is a fallback for cases where SCIP terminates too quickly to expose many
        branch nodes. Returned constraints are LP-relaxation compatible and KKT-valid
        once solved via linprog.
        """
        integer = init_node["integer"]
        bounds = init_node["bounds"]
        x_ref = np.asarray(x_ref, dtype=float)

        ranked = []
        for i, is_int in enumerate(integer):
            if not is_int:
                continue
            xi = float(x_ref[i])
            frac = abs(xi - round(xi))
            ranked.append((frac, i, xi))

        ranked.sort(reverse=True)

        cond_sets = []
        tol = 1e-7
        for _, i, xi in ranked[:max_vars]:
            lb, ub = bounds[i]

            lo = np.floor(xi)
            hi = np.ceil(xi)

            # Standard BnB split if the point is fractional.
            if abs(xi - round(xi)) > 1e-6:
                if lb is None or lo >= lb - tol:
                    cond_sets.append([(i, "<=", float(lo))])
                if ub is None or hi <= ub + tol:
                    cond_sets.append([(i, ">=", float(hi))])
                continue

            # If already integral, force a one-step move to create alternatives.
            xi_round = round(xi)
            down = xi_round - 1
            up = xi_round + 1
            if lb is None or down >= lb - tol:
                cond_sets.append([(i, "<=", float(down))])
            if ub is None or up <= ub + tol:
                cond_sets.append([(i, ">=", float(up))])

        return cond_sets

    def _fallback_lp_pool(self, init_node: LPNode) -> list[Candidate] | None:
        """LP-only pool for when SCIP itself fails (used in "mixed" mode)."""
        pool = _CandidatePool(_new_pool_debug())
        root_lp = solve_lp_with_conds(init_node, [], status="lp_fallback_root")
        pool.add(root_lp, "root_lp")

        if root_lp is not None:
            for conds in self._pseudo_branch_conds(init_node, root_lp["x"], max_vars=8):
                if len(pool) >= _MIN_MIXED_POOL_SIZE:
                    break
                pool.add(
                    solve_lp_with_conds(init_node, conds, status="lp_fallback_branch"),
                    "pseudo_lp",
                )

        return pool.results if pool.results else None

    def _leaves_pool(
        self, init_node: LPNode, tracker: NodeTracker, pool: _CandidatePool
    ) -> None:
        bounds = init_node["bounds"]
        leaf_node_ids = self._leaf_node_ids(tracker)
        pool_debug = pool.debug
        pool_debug.update(
            {
                "leaf_nodes_seen": 0,
                "deleted_nodes_seen": 0,
                "leaf_deleted_nodes_seen": 0,
                "leaf_lp_infeasible": 0,
                "leaf_invalid_candidate": 0,
            }
        )

        for node_num, info in tracker.node_map.items():
            if info.get("was_deleted", False):
                pool_debug["deleted_nodes_seen"] += 1
            if int(node_num) not in leaf_node_ids:
                continue

            pool_debug["leaf_nodes_seen"] += 1
            if info.get("was_deleted", False):
                pool_debug["leaf_deleted_nodes_seen"] += 1

            conds = list(info.get("conds", []))
            lp_solution = solve_lp_with_conds(init_node, conds)
            if lp_solution is None:
                pool_debug["leaf_lp_infeasible"] += 1
                continue

            x = np.asarray(lp_solution.get("x"), dtype=float)
            fun_value = float(lp_solution.get("fun"))
            if not self._is_valid_native_candidate(x, fun_value):
                pool_debug["leaf_invalid_candidate"] += 1
                continue

            pool.add(
                {
                    "fun": float(fun_value),
                    "x": x,
                    "eqlin": np.asarray(lp_solution.get("eqlin", []), dtype=float),
                    "ineqlin": np.asarray(lp_solution.get("ineqlin", []), dtype=float),
                    "lower": np.asarray(lp_solution.get("lower", []), dtype=float),
                    "upper": np.asarray(lp_solution.get("upper", []), dtype=float),
                    "fathomed": True,
                    "conds": conds,
                    "node": init_node,
                    "bounds": lp_solution.get("bounds", bounds),
                    "depth": info.get("depth"),
                    "node_num": node_num,
                    "parent": info.get("parent"),
                    "node_status": info.get("status"),
                    "branching_decision": info.get("branching_decision"),
                    "node_bounds": info.get("node_bounds"),
                    "reduced_costs": info.get("reduced_costs"),
                    "status": "scip_node_lp",
                },
                "leaf_lp",
            )

    def _mixed_pool(
        self,
        init_node: LPNode,
        model: Model,
        vars_: list[Variable],
        tracker: NodeTracker,
        ineq_cons_names: list[str],
        eq_cons_names: list[str],
        pool: _CandidatePool,
    ) -> None:
        c = np.asarray(init_node["c"], dtype=float)
        a_ub = init_node["A_ub"]
        a_eq = init_node["A_eq"]
        bounds = init_node["bounds"]
        n_vars = len(c)

        def _ordered_dual_array(
            names: list[str], duals_by_name: dict[str, float | None]
        ) -> np.ndarray:
            if not names:
                return np.array([], dtype=float)
            return np.array(
                [float(duals_by_name.get(n) or 0.0) for n in names], dtype=float
            )

        def _bound_multipliers(
            x: np.ndarray, ineqlin: np.ndarray, eqlin: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            resid = c.copy()
            if a_ub is not None and ineqlin is not None:
                resid = resid - (ineqlin @ a_ub)
            if a_eq is not None and eqlin is not None:
                resid = resid - (eqlin @ a_eq)
            lower = np.zeros(n_vars, dtype=float)
            upper = np.zeros(n_vars, dtype=float)
            tol = 1e-7
            for i, (lb, ub) in enumerate(bounds):
                xi = x[i]
                at_lb = (lb is not None) and abs(xi - lb) <= tol
                at_ub = (ub is not None) and abs(xi - ub) <= tol
                if at_lb and not at_ub:
                    lower[i] = resid[i]
                elif at_ub and not at_lb:
                    upper[i] = resid[i]
            return lower, upper

        # Root LP relaxation (KKT-valid, always provides diversity)
        root_lp = solve_lp_with_conds(init_node, [], status="lp_relaxation")
        pool.add(root_lp, "root_lp")

        # Best integer solution from SCIP
        best_sol = model.getBestSol()
        x_best = None
        if best_sol is not None:
            x_best = np.array([model.getVal(v) for v in vars_], dtype=float)
            d_ineq, d_eq = {}, {}
            for cons in model.getConss():
                name = cons.name
                if name not in eq_cons_names and name not in ineq_cons_names:
                    continue
                try:
                    val = float(model.getDualsolLinear(cons))
                except Warning:
                    # PySCIPOpt raises Warning when a dual value is unavailable after
                    # the MILP solve. The entry then fails the KKT check.
                    val = None
                if name in eq_cons_names:
                    d_eq[name] = val
                else:
                    d_ineq[name] = val
            ineqlin = _ordered_dual_array(ineq_cons_names, d_ineq)
            eqlin = _ordered_dual_array(eq_cons_names, d_eq)
            lower, upper = _bound_multipliers(x_best, ineqlin, eqlin)
            pool.add(
                {
                    "fun": float(model.getObjVal()),
                    "x": x_best,
                    "eqlin": eqlin,
                    "ineqlin": ineqlin,
                    "lower": lower,
                    "upper": upper,
                    "fathomed": False,
                    "conds": [],
                    "node": init_node,
                    "bounds": bounds,
                    "status": "optimal",
                },
                "optimal",
            )

        # LP relaxation along every branch path SCIP explored
        seen_paths = set()
        for info in tracker.node_map.values():
            conds = list(info.get("conds", []))
            if not conds or tuple(conds) in seen_paths:
                continue
            seen_paths.add(tuple(conds))
            pool.add(solve_lp_with_conds(init_node, conds), "branch_lp")

        # LP solutions SCIP found at the nodes it solved
        for node_num, info in tracker.node_map.items():
            lp_obj = info.get("lp_obj")
            primal_dict = info.get("primal", {})
            if not info.get("was_lp_solved") or lp_obj is None or not primal_dict:
                continue

            x = np.array(
                [primal_dict.get(f"x_{i}", 0.0) or 0.0 for i in range(n_vars)],
                dtype=float,
            )
            ineqlin = _ordered_dual_array(ineq_cons_names, info.get("duals_ineq", {}))
            eqlin = _ordered_dual_array(eq_cons_names, info.get("duals_eq", {}))
            lower, upper = _bound_multipliers(x, ineqlin, eqlin)
            pool.add(
                {
                    "fun": lp_obj,
                    "x": x,
                    "eqlin": eqlin,
                    "ineqlin": ineqlin,
                    "lower": lower,
                    "upper": upper,
                    "fathomed": bool(info.get("was_deleted", False)),
                    "conds": info.get("conds", []),
                    "node": init_node,
                    "bounds": bounds,
                    "depth": info.get("depth"),
                    "node_num": node_num,
                    "parent": info.get("parent"),
                    "node_status": info.get("status"),
                    "branching_decision": info.get("branching_decision"),
                    "node_bounds": info.get("node_bounds"),
                    "reduced_costs": info.get("reduced_costs"),
                    "status": "scip_tracked_lp",
                },
                "tracked",
            )

        # If SCIP yielded too few alternatives, synthesize a few LP candidates from
        # pseudo-branches around the best available solution.
        ref_x = x_best
        if ref_x is None and root_lp is not None:
            ref_x = root_lp["x"]
        if ref_x is not None:
            for conds in self._pseudo_branch_conds(init_node, ref_x, max_vars=8):
                if len(pool) >= _MIN_MIXED_POOL_SIZE:
                    break
                pool.add(
                    solve_lp_with_conds(init_node, conds, status="lp_pseudo_branch"),
                    "pseudo_lp",
                )

    @staticmethod
    def _has_valid_stationarity(
        init_node: LPNode, entry: Candidate, tol: float = 1e-4
    ) -> bool:
        x = entry.get("x")
        ineq = entry.get("ineqlin")
        eq = entry.get("eqlin")
        upper = entry.get("upper")
        lower = entry.get("lower")
        if x is None or ineq is None or eq is None or upper is None or lower is None:
            return False

        resid = np.asarray(init_node["c"], dtype=float).copy()
        if init_node["A_ub"] is not None:
            resid = resid - (np.asarray(ineq, dtype=float) @ init_node["A_ub"])
        if init_node["A_eq"] is not None:
            resid = resid - (np.asarray(eq, dtype=float) @ init_node["A_eq"])
        resid = resid - np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        return bool(np.all(np.abs(resid) <= tol))

    def solve(self, init_node: LPNode) -> list[Candidate] | None:
        self.last_tree_snapshot = None
        integer = init_node["integer"]
        bounds = init_node["bounds"]

        model = Model("MILP")
        try:
            branchrule = self._configure_model(model)
            vars_, ineq_cons_names, eq_cons_names = self._build_model(model, init_node)

            tracker = NodeTracker(
                integer, ineq_cons_names=ineq_cons_names, eq_cons_names=eq_cons_names
            )
            model.includeEventhdlr(
                tracker, "node_tracker", "collect visited node LP info"
            )

            optimize_error = None
            try:
                model.optimize()
            # PySCIPOpt reports SCIP errors as plain Exception, so nothing narrower works.
            except Exception as exc:  # noqa: BLE001
                optimize_error = exc

            # A bug in our own callbacks is not a solver failure, so always raise it.
            _raise_callback_errors(tracker, branchrule)

            if optimize_error is not None:
                if self.pool_mode == "mixed":
                    logger.warning(
                        "SCIP failed to solve the node; using the LP-only fallback pool.",
                        exc_info=optimize_error,
                    )
                    return self._fallback_lp_pool(init_node)
                logger.warning(
                    "SCIP failed to solve the node; returning no candidates.",
                    exc_info=optimize_error,
                )
                return None

            pool_debug = _new_pool_debug()
            if self.pool_mode == "leaves":
                # Leaf entries come straight from linprog, so they are KKT-valid.
                pool = _CandidatePool(pool_debug)
                self._leaves_pool(init_node, tracker, pool)
            else:
                # Mixed mode also uses SCIP's own duals, which need the KKT check.
                pool = _CandidatePool(
                    pool_debug,
                    check=lambda entry: self._has_valid_stationarity(init_node, entry),
                )
                self._mixed_pool(
                    init_node,
                    model,
                    vars_,
                    tracker,
                    ineq_cons_names,
                    eq_cons_names,
                    pool,
                )
            results = pool.results

            pool_debug["pre_filter_results"] = len(results)

            if self.tighten_integer_projected_bounds:
                tightened_results = []
                for entry in results:
                    tightened = dict(entry)
                    tightened["bounds"] = self._project_integer_bounds(
                        tightened["x"],
                        tightened.get("bounds", bounds),
                        integer,
                    )
                    tightened_results.append(tightened)
                results = tightened_results

            if self.mimic_bnb_pool_filter:
                pre_filter_count = len(results)
                results, filter_stats = self._filter_pool_like_bnb(results, integer)
                pool_debug["filter_removed"] = pre_filter_count - len(results)
                pool_debug["filter_fallback_to_ordered"] = int(
                    filter_stats.get("fallback_to_ordered", 0)
                )
                for key, value in filter_stats.items():
                    pool_debug[f"bnb_filter_{key}"] = value

            pool_debug["post_filter_results"] = len(results)

            self._store_tree_snapshot(
                tracker, results, str(model.getStatus()), pool_debug=pool_debug
            )

            if self.verbose:
                print(f"SCIP status: {model.getStatus()} (pool_mode={self.pool_mode})")
                print(
                    "Node events: "
                    f"focused={tracker.event_counts['NODEFOCUSED']} "
                    f"solved={tracker.event_counts['NODESOLVED']} "
                    f"lp_solved={tracker.event_counts['LPSOLVED']} "
                    f"deleted={tracker.event_counts['NODEDELETE']}"
                )
                print(
                    "LP obj extraction: "
                    f"ok={tracker.lp_obj_ok} "
                    f"invalid_lp_obj={tracker.skipped_invalid_lp_obj}"
                )
                print(f"Results returned: {len(results)}")
                print("Pool debug:", pool_debug)

            return results if results else None
        finally:
            self._cleanup_model(model)
