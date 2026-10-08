import numpy as np
from scipy.optimize import linprog


def normalize_linprog_duals(marginals, constraint_matrix):
    if constraint_matrix is None:
        return np.array([], dtype=float)

    duals = np.asarray(marginals, dtype=float)
    if duals.ndim == 0:
        n_constraints = constraint_matrix.shape[0]
        if n_constraints == 0:
            return np.array([], dtype=float)
        return duals.reshape(1)

    return duals.reshape(-1)


def var_name_to_id(var_name):
    """Map a variable name like "x_3" (or "t_x_3") to its index 3."""
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


def solve_lp_with_conds(init_node, conds, status="lp_branch_relaxation"):
    """Solve the LP relaxation of init_node with extra branching bounds.

    conds is a list of (var_id, op, value) with op ">=" or "<=". Returns a
    pool-compatible dict, or None if the bounds conflict or the LP is infeasible.
    """
    bounds = list(init_node["bounds"])

    for var_id, op, val in conds:
        idx = var_name_to_id(var_id)
        if not isinstance(idx, int) or idx < 0 or idx >= len(bounds):
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

    res = linprog(
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
        "eqlin": normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
        "ineqlin": normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
        "lower": np.asarray(res.lower.marginals, dtype=float),
        "upper": np.asarray(res.upper.marginals, dtype=float),
        "fathomed": False,
        "conds": list(conds),
        "node": init_node,
        "bounds": bounds,
        "status": status,
    }
