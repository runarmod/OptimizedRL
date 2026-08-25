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
from scipy.optimize import linprog as _linprog

_INVALID_OBJ = -1e18
_HEURISTICS = (
    "trivial",
    "feaspump",
    "rens",
    "rensub",
    "rounding",
    "simplerounding",
    "shifting",
    "fixandinvert",
    "oneopt",
    "trustregion",
    "ofins",
)


def _normalize_linprog_duals(marginals, constraint_matrix):
    if constraint_matrix is None:
        return np.array([], dtype=float)

    duals = np.asarray(marginals, dtype=float)
    if duals.ndim == 0:
        n_constraints = constraint_matrix.shape[0]
        if n_constraints == 0:
            return np.array([], dtype=float)
        return duals.reshape(1)

    return duals.reshape(-1)


def _var_name_to_id(var_name):
    if isinstance(var_name, int):
        return var_name
    if not isinstance(var_name, str):
        return var_name

    marker = "x_"
    idx = var_name.rfind(marker)
    if idx == -1:
        return var_name

    suffix = var_name[idx + len(marker) :]
    if suffix.isdigit():
        return int(suffix)
    return var_name


class NodeTracker(Eventhdlr):
    def __init__(self, integer, ineq_cons_names=None, eq_cons_names=None):
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
        self.skipped_no_lp_obj = 0
        self.skipped_invalid_lp_obj = 0
        self.lp_obj_ok = 0

    def eventinit(self):
        model = getattr(self, "model", None)
        if model is None:
            return
        for event_type in (
            SCIP_EVENTTYPE.NODEFOCUSED,
            SCIP_EVENTTYPE.NODESOLVED,
            SCIP_EVENTTYPE.NODEDELETE,
            SCIP_EVENTTYPE.LPSOLVED,
        ):
            try:
                model.catchEvent(event_type, self)
            except Exception:
                pass

    def eventinitsol(self):
        pass

    def eventexitsol(self):
        pass

    def eventexec(self, event):
        try:
            model = self.model
            event_type = event.getType()
        except Exception:
            return

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

    @staticmethod
    def _read_lp_primal(model, var):
        try:
            val = model.getSolVal(None, var)
            if val is not None:
                return float(val)
        except Exception:
            pass

        try:
            return float(model.getLPSolVal(var))
        except Exception:
            return None

    def _handle_node_event(self, model, event, status):
        node = None
        try:
            node = event.getNode()
        except Exception:
            node = None

        if node is None:
            try:
                node = model.getCurrentNode()
            except Exception:
                node = None

        if node is None:
            return

        try:
            node_num = node.getNumber()
        except Exception:
            return

        depth = None
        try:
            depth = node.getDepth()
        except Exception:
            pass

        parent_num = None
        try:
            parent = node.getParent()
            if parent is not None:
                parent_num = parent.getNumber()
        except Exception:
            pass

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

        lp_obj = None
        try:
            lp_obj = float(model.getLPObjVal())
        except Exception:
            pass

        if lp_obj is None:
            try:
                lp_obj = float(node.getLowerbound())
            except Exception:
                self.skipped_no_lp_obj += 1
                lp_obj = None

        if lp_obj is not None and lp_obj >= _INVALID_OBJ:
            self.lp_obj_ok += 1
            entry["lp_obj"] = lp_obj
        elif lp_obj is not None:
            self.skipped_invalid_lp_obj += 1

        local_conds = []
        try:
            branchings = node.getParentBranchings()
            if branchings is not None:
                vars_b, bounds_b, bound_types = branchings
                for var_obj, bound_val, bound_type in zip(
                    vars_b, bounds_b, bound_types
                ):
                    var_name = getattr(var_obj, "name", str(var_obj))
                    var_id = _var_name_to_id(var_name)
                    op = ">=" if int(bound_type) == 0 else "<="
                    local_conds.append((var_id, op, float(bound_val)))
                if local_conds:
                    last = local_conds[-1]
                    entry["branching_decision"] = {
                        "var": last[0],
                        "type": last[1],
                        "value": last[2],
                    }
        except Exception:
            local_conds = []

        try:
            node_bounds = {}
            for var in model.getVars():
                try:
                    lb_v = var.getLbLocal() if hasattr(var, "getLbLocal") else None
                    ub_v = var.getUbLocal() if hasattr(var, "getUbLocal") else None
                    node_bounds[var.name] = {"lb": lb_v, "ub": ub_v}
                except Exception:
                    pass
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
                    var_id = _var_name_to_id(var_name)
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
        except Exception:
            pass

        if status == "lp_solved":
            try:
                for var in model.getVars():
                    entry["primal"][var.name] = self._read_lp_primal(model, var)
            except Exception:
                pass

            try:
                for cons in model.getConss():
                    try:
                        name = model.getConsName(cons)
                    except Exception:
                        name = str(cons)
                    val = None
                    try:
                        val = float(model.getDualsolLinear(cons))
                    except Exception:
                        pass
                    if name in self.eq_cons_names:
                        entry["duals_eq"][name] = val
                    elif name in self.ineq_cons_names:
                        entry["duals_ineq"][name] = val
            except Exception:
                pass

            try:
                for var in model.getVars():
                    try:
                        entry["reduced_costs"][var.name] = float(
                            model.getVarRedcost(var)
                        )
                    except Exception:
                        entry["reduced_costs"][var.name] = None
            except Exception:
                pass

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

    def _handle_delete_event(self, event):
        try:
            node = event.getNode()
            if node is None:
                return
            node_num = node.getNumber()
        except Exception:
            return

        entry = self.node_map.get(node_num)
        if entry is None:
            return
        entry["was_deleted"] = True
        self.node_map[node_num] = entry


class BNBMostFractionalBranchrule(Branchrule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _bnb_fractionality(value):
        abs_value = abs(float(value))
        return abs_value - np.floor(abs_value)

    def branchexeclp(self, allowaddcons):
        try:
            branch_cands, branch_cand_sols, _, _, npriocands, _ = (
                self.model.getLPBranchCands()
            )
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if npriocands <= 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = None
        best_key = None
        for idx in range(npriocands):
            sol_value = float(branch_cand_sols[idx])
            frac = self._bnb_fractionality(sol_value)
            if frac < 1e-6:
                continue

            var_id = _var_name_to_id(
                getattr(branch_cands[idx], "name", str(branch_cands[idx]))
            )
            tie_break = int(var_id) if isinstance(var_id, int) else int(1e9)
            key = (frac, -tie_break)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        try:
            self.model.branchVarVal(
                branch_cands[best_idx], float(branch_cand_sols[best_idx])
            )
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        return {"result": SCIP_RESULT.BRANCHED}

    def branchexecext(self, branchcands, nbranchcands, npriobranchcands, allowaddcons):
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def branchexecps(self, allowaddcons):
        return {"result": SCIP_RESULT.DIDNOTRUN}


class SCIPSolver:
    def __init__(
        self,
        verbose=False,
        disable_heuristics=True,
        disable_presolve=True,
        disable_separating=True,
        disable_propagation=False,
        disable_conflict_analysis=False,
        disable_symmetry=False,
        prefer_most_fractional_branching=False,
        prefer_breadth_first=False,
        tighten_integer_projected_bounds=False,
        mimic_bnb_pool_filter=False,
        prefer_depth_first=True,
    ):
        self.verbose = verbose
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
    def _format_conds(conds):
        if not conds:
            return "root"
        return ", ".join(
            f"x_{_var_name_to_id(var_id)} {op} {float(val):.6g}"
            for var_id, op, val in conds
        )

    def _store_tree_snapshot(self, tracker, results, solve_status, pool_debug=None):
        pool_node_labels = []
        pool_node_label_set = set()

        def _append_pool_label(label):
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
            if (
                entry.get("status") == "scip_node_lp"
                and entry.get("node_num") is not None
            ):
                node_label = str(int(entry["node_num"]))
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

    def format_last_tree_log(self):
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
                branch_str = f"x_{_var_name_to_id(branch['var'])} {branch['type']} {float(branch['value']):.6g}"
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
    def _cleanup_model(model):
        try:
            stage_name = model.getStageName()
        except Exception:
            return

        if stage_name == "SOLVED":
            try:
                model.freeTransform()
            except Exception:
                pass
            try:
                stage_name = model.getStageName()
            except Exception:
                return

        if stage_name == "PROBLEM":
            try:
                model.freeProb()
            except Exception:
                pass

    @staticmethod
    def _safe_set(model, param, value):
        try:
            model.setParam(param, value)
        except Exception:
            pass

    def _configure_model(self, model):
        if not self.verbose:
            model.hideOutput()

        if self.disable_presolve:
            try:
                model.setPresolve(SCIP_PARAMSETTING.OFF)
            except Exception:
                self._safe_set(model, "presolving/maxrounds", 0)

        if self.disable_separating:
            try:
                model.setSeparating(SCIP_PARAMSETTING.OFF)
            except Exception:
                self._safe_set(model, "separating/maxrounds", 0)
                self._safe_set(model, "separating/maxroundsroot", 0)

        if self.disable_heuristics:
            try:
                model.setHeuristics(SCIP_PARAMSETTING.OFF)
            except Exception:
                for heuristic in _HEURISTICS:
                    self._safe_set(model, f"heuristics/{heuristic}/freq", -1)

        if self.disable_propagation:
            try:
                model.disablePropagation()
            except Exception:
                pass

        if self.disable_conflict_analysis:
            self._safe_set(model, "conflict/enable", False)

        if self.disable_symmetry:
            self._safe_set(model, "misc/usesymmetry", 0)

        if self.prefer_most_fractional_branching:
            try:
                model.includeBranchrule(
                    BNBMostFractionalBranchrule(),
                    "bnb_most_fractional",
                    "branch like bnb.py on the variable with largest fractional part",
                    priority=10_000_000,
                    maxdepth=-1,
                    maxbounddist=1,
                )
            except Exception:
                pass
            self._safe_set(model, "branching/leastinf/priority", 1000000)
            self._safe_set(model, "branching/pscost/priority", -1000000)
            self._safe_set(model, "branching/relpscost/priority", -1000000)
            self._safe_set(model, "branching/inference/priority", -1000000)

        if self.prefer_breadth_first:
            self._safe_set(model, "nodeselection/bfs/stdpriority", 1000000)
            self._safe_set(model, "nodeselection/dfs/stdpriority", -1000000)
            self._safe_set(model, "nodeselection/estimate/stdpriority", -1000000)
            self._safe_set(model, "nodeselection/hybridestim/stdpriority", -1000000)
        elif self.prefer_depth_first:
            self._safe_set(model, "nodeselection/dfs/stdpriority", 1000000)

    @staticmethod
    def _project_integer_bounds(x, bounds, integer):
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
    def _all_integer(x, integer, tol=1e-6):
        x = np.asarray(x, dtype=float)
        return all(
            (abs(float(value) - round(float(value))) < tol) if is_integer else True
            for value, is_integer in zip(x, integer)
        )

    @staticmethod
    def _bnb_like_order_key(entry):
        depth = entry.get("depth")
        node_num = entry.get("node_num")
        return (
            int(depth) if depth is not None else -1,
            int(node_num) if node_num is not None else 10**9,
        )

    def _filter_pool_like_bnb(self, results, integer):
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
    def _is_valid_native_candidate(x, fun_value):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            return False
        if np.any(np.abs(x) >= 1e19):
            return False
        if not np.isfinite(fun_value):
            return False
        return not fun_value <= _INVALID_OBJ

    @staticmethod
    def _leaf_node_ids(tracker):
        child_counts = {}
        for node_num in tracker.node_map:
            child_counts[int(node_num)] = 0

        for node_num, info in tracker.node_map.items():
            parent = info.get("parent")
            if parent is None:
                continue
            try:
                parent_num = int(parent)
            except Exception:
                continue
            child_counts[parent_num] = child_counts.get(parent_num, 0) + 1

        return {int(node_num) for node_num, count in child_counts.items() if count == 0}

    def _solve_lp_with_conds(self, init_node, conds):
        bounds = list(init_node["bounds"])

        def _to_index(var_id):
            parsed = _var_name_to_id(var_id)
            return parsed if isinstance(parsed, int) else None

        for var_id, op, val in conds:
            idx = _to_index(var_id)
            if idx is None or idx < 0 or idx >= len(bounds):
                continue
            lb, ub = bounds[idx]
            if op == ">=":
                lb = float(val) if lb is None else max(float(lb), float(val))
            elif op == "<=":
                ub = float(val) if ub is None else min(float(ub), float(val))
            bounds[idx] = (lb, ub)

        for lb, ub in bounds:
            if lb is not None and ub is not None and lb > ub + 1e-10:
                return None

        res = _linprog(
            init_node["c"],
            A_ub=init_node["A_ub"],
            b_ub=init_node["b_ub"],
            A_eq=init_node["A_eq"],
            b_eq=init_node["b_eq"],
            bounds=bounds,
        )
        if not res.success:
            return None

        return {
            "fun": float(res.fun),
            "x": np.asarray(res.x, dtype=float),
            "eqlin": _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
            "ineqlin": _normalize_linprog_duals(
                res.ineqlin.marginals, init_node["A_ub"]
            ),
            "lower": np.asarray(res.lower.marginals, dtype=float),
            "upper": np.asarray(res.upper.marginals, dtype=float),
            "fathomed": False,
            "conds": list(conds),
            "node": init_node,
            "bounds": bounds,
            "status": "lp_branch_relaxation",
        }

    def solve(self, init_node):
        self.last_tree_snapshot = None
        c = init_node["c"]
        a_ub = init_node["A_ub"]
        b_ub = init_node["b_ub"]
        a_eq = init_node["A_eq"]
        b_eq = init_node["b_eq"]
        bounds = init_node["bounds"]
        integer = init_node["integer"]
        n_vars = len(c)

        model = Model("MILP")
        try:
            self._configure_model(model)

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
                    try:
                        ineq_cons_names.append(model.getConsName(cons))
                    except Exception:
                        ineq_cons_names.append(str(cons))

            eq_cons_names = []
            if a_eq is not None and b_eq is not None:
                for row_idx in range(a_eq.shape[0]):
                    expr = quicksum(
                        float(a_eq[row_idx, col_idx]) * vars_[col_idx]
                        for col_idx in range(n_vars)
                        if a_eq[row_idx, col_idx] != 0
                    )
                    cons = model.addCons(expr == float(b_eq[row_idx]))
                    try:
                        eq_cons_names.append(model.getConsName(cons))
                    except Exception:
                        eq_cons_names.append(str(cons))

            tracker = NodeTracker(
                integer, ineq_cons_names=ineq_cons_names, eq_cons_names=eq_cons_names
            )
            try:
                model.includeEventhdlr(
                    tracker, "node_tracker", "collect visited node LP info"
                )
            except Exception:
                pass

            try:
                model.optimize()
            except Exception:
                return None

            results = []
            seen_x = set()
            added_nodes = 0
            leaf_node_ids = self._leaf_node_ids(tracker)
            pool_debug = {
                "leaf_nodes_seen": 0,
                "deleted_nodes_seen": 0,
                "leaf_deleted_nodes_seen": 0,
                "leaf_lp_infeasible": 0,
                "leaf_invalid_candidate": 0,
                "duplicate_candidates": 0,
                "pre_filter_results": 0,
                "post_filter_results": 0,
                "filter_removed": 0,
                "filter_fallback_to_ordered": 0,
            }

            def _append_unique(entry):
                x = entry.get("x")
                if x is None:
                    return False
                signature = tuple(np.round(np.asarray(x, dtype=float), 8).tolist())
                if signature in seen_x:
                    pool_debug["duplicate_candidates"] += 1
                    return False
                seen_x.add(signature)
                results.append(entry)
                return True

            for node_num, info in tracker.node_map.items():
                if info.get("was_deleted", False):
                    pool_debug["deleted_nodes_seen"] += 1
                if int(node_num) not in leaf_node_ids:
                    continue

                pool_debug["leaf_nodes_seen"] += 1
                if info.get("was_deleted", False):
                    pool_debug["leaf_deleted_nodes_seen"] += 1

                conds = list(info.get("conds", []))
                lp_solution = self._solve_lp_with_conds(init_node, conds)
                if lp_solution is None:
                    pool_debug["leaf_lp_infeasible"] += 1
                    continue

                x = np.asarray(lp_solution.get("x"), dtype=float)
                fun_value = float(lp_solution.get("fun"))
                if not self._is_valid_native_candidate(x, fun_value):
                    pool_debug["leaf_invalid_candidate"] += 1
                    continue

                if _append_unique(
                    {
                        "fun": float(fun_value),
                        "x": x,
                        "eqlin": np.asarray(lp_solution.get("eqlin", []), dtype=float),
                        "ineqlin": np.asarray(
                            lp_solution.get("ineqlin", []), dtype=float
                        ),
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
                    }
                ):
                    added_nodes += 1

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
                print(f"SCIP status: {model.getStatus()}")
                print(
                    "Node events: "
                    f"focused={tracker.event_counts['NODEFOCUSED']} "
                    f"solved={tracker.event_counts['NODESOLVED']} "
                    f"deleted={tracker.event_counts['NODEDELETE']}"
                )
                print(
                    "LP obj extraction: "
                    f"ok={tracker.lp_obj_ok} "
                    f"no_lp_obj={tracker.skipped_no_lp_obj} "
                    f"invalid_lp_obj={tracker.skipped_invalid_lp_obj}"
                )
                print(f"Visited SCIP nodes added: {added_nodes}")
                print(f"Results returned: {len(results)}")
                print("Pool debug:", pool_debug)

            return results if results else None
        finally:
            self._cleanup_model(model)
