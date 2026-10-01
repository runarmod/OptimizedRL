import numpy as np

from src.dap.simulator import all_assortments
from src.solvers.solver_interface import Solver


class ExactAssortmentSolver(Solver):
    """Every feasible assortment as a candidate: the exact CORL policy.

    For an assortment MILP (``node["assortment"]`` = {"n_items", "k"};
    variables [x_1..x_N, v], value pieces as rows ``slope_j . x - v <= rhs_j``)
    the objective of a fixed assortment has a closed form,
    Q(x) = c_x . x + max_j(slope_j . x - rhs_j), so the softmax over all
    actions in CORL eq. (10) can be computed exactly, without the B&B
    approximation of eqs. (12)-(14). The subgradient uses the active piece
    (dual -1 in scipy's sign convention).

    ``max_candidates`` keeps only the lowest-Q assortments, which bounds
    memory; their softmax mass dominates when Q values are spread out.
    """

    def __init__(self, max_candidates: int | None = None):
        self.max_candidates = max_candidates
        self._cache = {}

    def solve(self, node):
        spec = node["assortment"]
        n, k = spec["n_items"], spec["k"]
        if (n, k) not in self._cache:
            self._cache[(n, k)] = all_assortments(n, k)
        assortments = self._cache[(n, k)]
        c = np.asarray(node["c"], dtype=float)
        values = assortments @ c[:n]
        has_v = len(c) > n
        if has_v:
            A_ub = np.asarray(node["A_ub"], dtype=float)
            piece = assortments @ A_ub[:, :n].T - np.asarray(node["b_ub"], dtype=float)
            active = piece.argmax(axis=1)
            v = piece[np.arange(len(assortments)), active]
            values = values + c[n] * v
            n_rows = A_ub.shape[0]

        order = np.argsort(values, kind="stable")
        if self.max_candidates is not None:
            order = order[: self.max_candidates]
        pool = []
        for m in order:
            ineqlin = np.zeros(n_rows if has_v else 0)
            if has_v:
                ineqlin[active[m]] = -1.0
            x = np.append(assortments[m], v[m]) if has_v else assortments[m].copy()
            pool.append({
                "fun": float(values[m]),
                "x": x,
                "ineqlin": ineqlin,
                "eqlin": np.zeros(1),
                "fathomed": False,
                "conds": [],
                "node": node,
                "bounds": node["bounds"],
                "status": "assortment",
            })
        return pool
