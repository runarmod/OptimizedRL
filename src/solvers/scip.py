import numpy as np
from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model, quicksum
from scipy.optimize import linprog as _linprog

_INVALID_OBJ = -1e18  # threshold below which an LP objective is considered garbage


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


class NodeTracker(Eventhdlr):
    """Collects per-node LP primal/dual solutions via NODEFOCUSED events.

    Only nodes where SCIP has actually solved the LP relaxation (i.e. getLPObjval()
    returns a finite value) are recorded.  This requires SCIP to be configured with
    presolving, cuts, and heuristics disabled (done in SCIPSolver.solve()).
    """

    def __init__(self, integer, ineq_cons_names=None, eq_cons_names=None):
        super().__init__()
        self.integer = integer
        self.ineq_cons_names = list(ineq_cons_names or [])
        self.eq_cons_names = list(eq_cons_names or [])
        self.node_map = {}       # node_num -> entry dict
        self.parent_bounds = {}  # node_num -> {var_name: (lb, ub)}
        self.event_counts = {
            "NODEFOCUSED": 0,
            "NODESOLVED": 0,
            "NODEDELETE": 0,
        }
        self.branch_paths = []
        self._branch_path_signatures = set()
        self.skipped_no_lp_obj = 0
        self.skipped_invalid_lp_obj = 0
        self.lp_obj_ok = 0

    def eventinit(self):
        m = getattr(self, 'model', None)
        if m is None:
            return
        for etype in (SCIP_EVENTTYPE.NODEFOCUSED, SCIP_EVENTTYPE.NODESOLVED, SCIP_EVENTTYPE.NODEDELETE):
            try:
                m.catchEvent(etype, self)
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

        if event_type in (SCIP_EVENTTYPE.NODEFOCUSED, SCIP_EVENTTYPE.NODESOLVED):
            if event_type == SCIP_EVENTTYPE.NODEFOCUSED:
                self.event_counts["NODEFOCUSED"] += 1
            else:
                self.event_counts["NODESOLVED"] += 1
            self._handle_node_event(model, event)
        elif event_type == SCIP_EVENTTYPE.NODEDELETE:
            self.event_counts["NODEDELETE"] += 1
            self._handle_delete_event(event)

    def _handle_node_event(self, model, event):
        try:
            node = event.getNode()
            if node is None:
                return
            node_num = node.getNumber()
        except Exception:
            return

        # Capture branch conditions even if LP objective is unavailable.
        try:
            branchings = node.getParentBranchings()
            if branchings is not None:
                vars_b, bounds_b, bound_types = branchings
                conds = []
                for var_obj, bound_val, btype in zip(vars_b, bounds_b, bound_types):
                    var_name = getattr(var_obj, 'name', str(var_obj))
                    var_id = var_name
                    if isinstance(var_name, str) and var_name.startswith("x_"):
                        try:
                            var_id = int(var_name.split("_", 1)[1])
                        except Exception:
                            pass
                    op = '>=' if int(btype) == 0 else '<='
                    conds.append((var_id, op, float(bound_val)))
                if conds:
                    sig = tuple(conds)
                    if sig not in self._branch_path_signatures:
                        self._branch_path_signatures.add(sig)
                        self.branch_paths.append(conds)
        except Exception:
            pass

        # KEY FIX: use getLPObjval() — the actual LP relaxation objective at this node.
        # getLowerbound() returns the dual bound which can be -1e20 for unsolved nodes.
        lp_obj = None
        try:
            lp_obj = float(model.getLPObjval())
        except Exception:
            self.skipped_no_lp_obj += 1
        if lp_obj is None:
            return  # no valid LP was solved here; skip
        if lp_obj < _INVALID_OBJ:
            self.skipped_invalid_lp_obj += 1
            return  # no valid LP was solved here; skip
        self.lp_obj_ok += 1

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
            'node_num': node_num,
            'depth': depth,
            'parent': parent_num,
            'lp_obj': lp_obj,
            'status': 'focused',
            'branching_decision': None,
            'node_bounds': {},
            'conds': [],
            'primal': {},
            'duals_ineq': {},
            'duals_eq': {},
            'reduced_costs': {},
        }

        # --- Branch-path conditions ---
        local_conds = []
        try:
            branchings = node.getParentBranchings()
            if branchings is not None:
                vars_b, bounds_b, bound_types = branchings
                for var_obj, bound_val, btype in zip(vars_b, bounds_b, bound_types):
                    var_name = getattr(var_obj, 'name', str(var_obj))
                    var_id = var_name
                    if isinstance(var_name, str) and var_name.startswith("x_"):
                        try:
                            var_id = int(var_name.split("_", 1)[1])
                        except Exception:
                            pass
                    op = '>=' if int(btype) == 0 else '<='
                    local_conds.append((var_id, op, float(bound_val)))
                if local_conds:
                    last = local_conds[-1]
                    entry['branching_decision'] = {'var': last[0], 'type': last[1], 'value': last[2]}
        except Exception:
            local_conds = []

        # --- Node-local variable bounds (for fallback branching deduction) ---
        try:
            node_bounds = {}
            for v in model.getVars():
                try:
                    lb_v = v.getLbLocal() if hasattr(v, 'getLbLocal') else None
                    ub_v = v.getUbLocal() if hasattr(v, 'getUbLocal') else None
                    node_bounds[v.name] = {'lb': lb_v, 'ub': ub_v}
                except Exception:
                    pass
            entry['node_bounds'] = node_bounds

            if node_bounds and not local_conds and parent_num is not None and parent_num in self.parent_bounds:
                for var_name, binfo in node_bounds.items():
                    plb, pub = self.parent_bounds[parent_num].get(var_name, (None, None))
                    clb, cub = binfo.get('lb'), binfo.get('ub')
                    vid = var_name
                    if isinstance(var_name, str) and var_name.startswith("x_"):
                        try:
                            vid = int(var_name.split("_", 1)[1])
                        except Exception:
                            pass
                    if clb is not None and plb is not None and clb > plb + 1e-6:
                        entry['branching_decision'] = {'var': vid, 'type': '>=', 'value': clb}
                        local_conds = [(vid, '>=', clb)]
                        break
                    if cub is not None and pub is not None and cub < pub - 1e-6:
                        entry['branching_decision'] = {'var': vid, 'type': '<=', 'value': cub}
                        local_conds = [(vid, '<=', cub)]
                        break

            conds = list(self.node_map[parent_num].get('conds', [])) if parent_num in self.node_map else []
            conds.extend(local_conds)
            entry['conds'] = conds

            self.parent_bounds[node_num] = {
                v: (binfo.get('lb'), binfo.get('ub')) for v, binfo in node_bounds.items()
            }
        except Exception:
            pass

        # --- LP primal solution ---
        try:
            for v in model.getVars():
                try:
                    entry['primal'][v.name] = float(model.getLPSolVal(v))
                except Exception:
                    entry['primal'][v.name] = None
        except Exception:
            pass

        # --- Dual values ---
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
                    entry['duals_eq'][name] = val
                elif name in self.ineq_cons_names:
                    entry['duals_ineq'][name] = val
        except Exception:
            pass

        # --- Reduced costs ---
        try:
            for v in model.getVars():
                try:
                    entry['reduced_costs'][v.name] = float(model.getVarRedcost(v))
                except Exception:
                    entry['reduced_costs'][v.name] = None
        except Exception:
            pass

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
        entry['status'] = 'deleted'
        self.node_map[node_num] = entry


class SCIPSolver:
    """SCIP-based MILP solver. Uses SCIP's native B&B engine with LP relaxations tracked
    via event handlers. SCIP manages branching, cutting planes, and primal heuristics —
    so it scales to larger problems naturally. Only nodes where a valid LP objective was
    obtained (getLPObjval() >= _INVALID_OBJ) are included in the returned pool."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def _cleanup_model(model):
        try:
            stage_name = model.getStageName()
        except Exception:
            return

        # After a normal solve, free the transformed problem first.
        if stage_name == "SOLVED":
            try:
                model.freeTransform()
            except Exception:
                pass
            try:
                stage_name = model.getStageName()
            except Exception:
                return

        # Free the original problem only when SCIP is back in PROBLEM stage.
        if stage_name == "PROBLEM":
            try:
                model.freeProb()
            except Exception:
                pass

    @staticmethod
    def _safe_set(model, param, val):
        try:
            model.setParam(param, val)
        except Exception:
            pass

    def _solve_root_lp(self, init_node):
        """Solve LP relaxation (all vars continuous) via scipy linprog.
        The LP solution satisfies KKT (dLdx=0), required by actor.py.
        Returns a pool-compatible dict or None if infeasible.
        """
        res = _linprog(
            init_node["c"],
            A_ub=init_node["A_ub"], b_ub=init_node["b_ub"],
            A_eq=init_node["A_eq"], b_eq=init_node["b_eq"],
            bounds=init_node["bounds"],
        )
        if not res.success:
            return None
        x = np.asarray(res.x, dtype=float)
        ineqlin = _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"])
        eqlin = _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"])
        lower = np.asarray(res.lower.marginals, dtype=float)
        upper = np.asarray(res.upper.marginals, dtype=float)
        return {
            'fun': float(res.fun),
            'x': x,
            'eqlin': eqlin,
            'ineqlin': ineqlin,
            'lower': lower,
            'upper': upper,
            'fathomed': False,
            'conds': [],
            'node': init_node,
            'bounds': init_node["bounds"],
            'status': 'lp_relaxation',
        }

    def _solve_lp_with_conds(self, init_node, conds):
        bounds = list(init_node["bounds"])

        def _to_index(var_id):
            if isinstance(var_id, int):
                return var_id
            if isinstance(var_id, str) and var_id.startswith("x_"):
                try:
                    return int(var_id.split("_", 1)[1])
                except Exception:
                    return None
            return None

        for var_id, op, val in conds:
            idx = _to_index(var_id)
            if idx is None or idx < 0 or idx >= len(bounds):
                continue
            lb, ub = bounds[idx]
            if op == '>=':
                lb = float(val) if lb is None else max(float(lb), float(val))
            elif op == '<=':
                ub = float(val) if ub is None else min(float(ub), float(val))
            bounds[idx] = (lb, ub)

        for lb, ub in bounds:
            if lb is not None and ub is not None and lb > ub + 1e-10:
                return None

        res = _linprog(
            init_node["c"],
            A_ub=init_node["A_ub"], b_ub=init_node["b_ub"],
            A_eq=init_node["A_eq"], b_eq=init_node["b_eq"],
            bounds=bounds,
        )
        if not res.success:
            return None

        x = np.asarray(res.x, dtype=float)
        ineqlin = _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"])
        eqlin = _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"])
        lower = np.asarray(res.lower.marginals, dtype=float)
        upper = np.asarray(res.upper.marginals, dtype=float)
        return {
            'fun': float(res.fun),
            'x': x,
            'eqlin': eqlin,
            'ineqlin': ineqlin,
            'lower': lower,
            'upper': upper,
            'fathomed': False,
            'conds': list(conds),
            'node': init_node,
            'bounds': bounds,
            'status': 'lp_branch_relaxation',
        }

    def _pseudo_branch_conds(self, init_node, x_ref, max_vars=6):
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
            xi_round = int(round(xi))
            down = xi_round - 1
            up = xi_round + 1
            if lb is None or down >= lb - tol:
                cond_sets.append([(i, "<=", float(down))])
            if ub is None or up <= ub + tol:
                cond_sets.append([(i, ">=", float(up))])

        return cond_sets

    def _fallback_lp_pool(self, init_node, root_lp=None, min_pool_size=4):
        results = []
        seen_x = set()

        def _append_unique(entry):
            if entry is None or entry.get('x') is None:
                return False
            sig = tuple(np.round(np.asarray(entry['x'], dtype=float), 8).tolist())
            if sig in seen_x:
                return False
            seen_x.add(sig)
            results.append(entry)
            return True

        if root_lp is None:
            root_lp = self._solve_root_lp(init_node)
        if root_lp is not None:
            root_lp = dict(root_lp)
            root_lp['status'] = 'lp_fallback_root'
            _append_unique(root_lp)

        ref_x = None if root_lp is None else np.asarray(root_lp['x'], dtype=float)
        if ref_x is not None:
            for conds in self._pseudo_branch_conds(init_node, ref_x, max_vars=8):
                if len(results) >= min_pool_size:
                    break
                lp_alt = self._solve_lp_with_conds(init_node, conds)
                if lp_alt is None:
                    continue
                lp_alt['status'] = 'lp_fallback_branch'
                _append_unique(lp_alt)

        return results if results else None

    def solve(self, init_node):
        c = init_node["c"]
        A_ub = init_node["A_ub"]
        b_ub = init_node["b_ub"]
        A_eq = init_node["A_eq"]
        b_eq = init_node["b_eq"]
        bounds = init_node["bounds"]
        integer = init_node["integer"]
        n_vars = len(c)

        model = Model("MILP")
        try:
            # --- Root LP relaxation (always in pool; guarantees KKT-valid candidates) ---
            root_lp = self._solve_root_lp(init_node)

            if not self.verbose:
                model.hideOutput()

            # Disable presolving/heuristics/cuts so SCIP solves each node's LP relaxation
            # and dual values / LP primal solutions are accessible in the event handler.
            self._safe_set(model, "presolving/maxrounds", 0)
            self._safe_set(model, "separating/maxrounds", 0)
            self._safe_set(model, "separating/maxroundsroot", 0)
            for h in ("feaspump", "rens", "rensub", "rounding", "simplerounding",
                      "shifting", "fixandinvert", "oneopt", "trustregion", "ofins"):
                self._safe_set(model, f"heuristics/{h}/freq", -1)
            # Depth-first ensures LP is solved completely at each node before branching.
            self._safe_set(model, "nodesel/dfs/stdpriority", 1000000)

            # --- Variables ---
            vars_ = []
            for i in range(n_vars):
                lb, ub = bounds[i]
                vars_.append(model.addVar(
                    name=f"x_{i}",
                    lb=float(lb) if lb is not None else -1e20,
                    ub=float(ub) if ub is not None else 1e20,
                    vtype='I' if integer[i] else 'C',
                ))

            model.setObjective(quicksum(float(c[i]) * vars_[i] for i in range(n_vars)), sense='minimize')

            # --- Constraints ---
            ineq_cons_names = []
            if A_ub is not None and b_ub is not None:
                for i in range(A_ub.shape[0]):
                    expr = quicksum(float(A_ub[i, j]) * vars_[j] for j in range(n_vars) if A_ub[i, j] != 0)
                    cons = model.addCons(expr <= float(b_ub[i]))
                    try:
                        ineq_cons_names.append(model.getConsName(cons))
                    except Exception:
                        ineq_cons_names.append(str(cons))

            eq_cons_names = []
            if A_eq is not None and b_eq is not None:
                for i in range(A_eq.shape[0]):
                    expr = quicksum(float(A_eq[i, j]) * vars_[j] for j in range(n_vars) if A_eq[i, j] != 0)
                    cons = model.addCons(expr == float(b_eq[i]))
                    try:
                        eq_cons_names.append(model.getConsName(cons))
                    except Exception:
                        eq_cons_names.append(str(cons))

            # --- Attach event handler ---
            tracker = NodeTracker(integer, ineq_cons_names=ineq_cons_names, eq_cons_names=eq_cons_names)
            try:
                model.includeEventhdlr(tracker, "node_tracker", "collect node LP info")
            except Exception:
                pass

            try:
                model.optimize()
            except Exception:
                return self._fallback_lp_pool(init_node, root_lp=root_lp)

            # --- Build result pool ---
            def _ordered_dual_array(names, duals_by_name):
                if not names:
                    return np.array([], dtype=float)
                return np.array([float(duals_by_name.get(n) or 0.0) for n in names], dtype=float)

            def _bound_multipliers(x, ineqlin, eqlin):
                resid = np.asarray(c, dtype=float).copy()
                if A_ub is not None and ineqlin is not None:
                    resid = resid - (ineqlin @ A_ub)
                if A_eq is not None and eqlin is not None:
                    resid = resid - (eqlin @ A_eq)
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

            def _has_valid_stationarity(entry, tol=1e-4):
                x = entry.get('x')
                ineq = entry.get('ineqlin')
                eq = entry.get('eqlin')
                upper = entry.get('upper')
                lower = entry.get('lower')
                if x is None or ineq is None or eq is None or upper is None or lower is None:
                    return False

                resid = np.asarray(c, dtype=float).copy()
                if A_ub is not None:
                    resid = resid - (np.asarray(ineq, dtype=float) @ A_ub)
                if A_eq is not None:
                    resid = resid - (np.asarray(eq, dtype=float) @ A_eq)
                resid = resid - np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
                return bool(np.all(np.abs(resid) <= tol))

            results = []
            seen_x = set()

            def _append_unique(entry):
                x = entry.get('x')
                if x is None:
                    return False
                if not _has_valid_stationarity(entry):
                    return False
                sig = tuple(np.round(np.asarray(x, dtype=float), 8).tolist())
                if sig in seen_x:
                    return False
                seen_x.add(sig)
                results.append(entry)
                return True

            added_root_lp = 0
            added_optimal = 0
            added_tracked = 0
            added_branch_lp = 0
            added_pseudo_lp = 0

            # Start pool with root LP relaxation (KKT-valid, always provides diversity)
            if root_lp is not None:
                if _append_unique(root_lp):
                    added_root_lp = 1

            # Best integer solution from SCIP
            best_sol = model.getBestSol()
            if best_sol is not None:
                x_opt = np.array([model.getVal(v) for v in vars_], dtype=float)
                d_ineq, d_eq = {}, {}
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
                        if name in eq_cons_names:
                            d_eq[name] = val
                        elif name in ineq_cons_names:
                            d_ineq[name] = val
                except Exception:
                    pass
                ineqlin = _ordered_dual_array(ineq_cons_names, d_ineq)
                eqlin = _ordered_dual_array(eq_cons_names, d_eq)
                lower, upper = _bound_multipliers(x_opt, ineqlin, eqlin)
                if _append_unique({
                    'fun': float(model.getObjVal()),
                    'x': x_opt,
                    'eqlin': eqlin,
                    'ineqlin': ineqlin,
                    'lower': lower,
                    'upper': upper,
                    'fathomed': False,
                    'conds': [],
                    'node': init_node,
                    'bounds': bounds,
                    'status': 'optimal',
                }):
                    added_optimal = 1

            # SCIP-informed branch LP relaxations reconstructed from event branch conditions.
            for conds in tracker.branch_paths:
                lp_branch = self._solve_lp_with_conds(init_node, conds)
                if lp_branch is None:
                    continue
                if _append_unique(lp_branch):
                    added_branch_lp += 1

            # Tracked B&B nodes — only those with valid LP objectives
            for node_num, info in tracker.node_map.items():
                lp_obj = info.get('lp_obj')
                if lp_obj is None or lp_obj < _INVALID_OBJ:
                    continue

                primal_dict = info.get('primal', {})
                x = None
                if primal_dict:
                    x = np.array([primal_dict.get(f"x_{i}", 0.0) or 0.0 for i in range(n_vars)], dtype=float)

                ineqlin = _ordered_dual_array(ineq_cons_names, info.get('duals_ineq', {}))
                eqlin = _ordered_dual_array(eq_cons_names, info.get('duals_eq', {}))
                lower, upper = (None, None)
                if x is not None:
                    lower, upper = _bound_multipliers(x, ineqlin, eqlin)

                if _append_unique({
                    'fun': lp_obj,
                    'x': x,
                    'eqlin': eqlin,
                    'ineqlin': ineqlin,
                    'lower': lower,
                    'upper': upper,
                    'fathomed': info.get('status') == 'deleted',
                    'conds': info.get('conds', []),
                    'node': init_node,
                    'bounds': bounds,
                    'depth': info.get('depth'),
                    'node_num': node_num,
                    'parent': info.get('parent'),
                    'node_status': info.get('status'),
                    'branching_decision': info.get('branching_decision'),
                    'node_bounds': info.get('node_bounds'),
                    'reduced_costs': info.get('reduced_costs'),
                }):
                    added_tracked += 1

            # One fallback fix: if SCIP yielded too few alternatives, synthesize a few
            # LP candidates from pseudo-branches around the best available solution.
            min_pool_size = 4
            if len(results) < min_pool_size:
                ref_x = None
                if best_sol is not None:
                    ref_x = np.array([model.getVal(v) for v in vars_], dtype=float)
                elif root_lp is not None:
                    ref_x = np.asarray(root_lp["x"], dtype=float)

                if ref_x is not None:
                    for conds in self._pseudo_branch_conds(init_node, ref_x, max_vars=8):
                        if len(results) >= min_pool_size:
                            break
                        lp_alt = self._solve_lp_with_conds(init_node, conds)
                        if lp_alt is None:
                            continue
                        lp_alt["status"] = "lp_pseudo_branch"
                        if _append_unique(lp_alt):
                            added_pseudo_lp += 1

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
                print(f"Nodes tracked: {len(tracker.node_map)}")
                print(
                    "Pool composition: "
                    f"root_lp={added_root_lp} optimal={added_optimal} "
                    f"branch_lp={added_branch_lp} tracked={added_tracked} "
                    f"pseudo_lp={added_pseudo_lp}"
                )
                print(f"Results returned: {len(results)}")

            return results
        finally:
            self._cleanup_model(model)


