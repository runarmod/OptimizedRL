import numpy as np

from src.dap.simulator import all_assortments
from src.solvers.solver_interface import Solver


class ExactAssortmentSolver(Solver):
    """Every feasible assortment as a candidate: the exact CORL policy.

    For an assortment MILP the objective of a fixed assortment has a closed
    form, which the model provides as ``node["assortment"]["evaluate"]``
    (returning Q, the full variable vectors and the value-piece duals). The
    softmax over all actions in CORL eq. (10) can then be computed exactly,
    without the B&B approximation of eqs. (12)-(14).

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
        values, xs, duals = spec["evaluate"](self._cache[(n, k)])
        order = np.argsort(values, kind="stable")
        if self.max_candidates is not None:
            order = order[: self.max_candidates]
        return [
            {
                "fun": float(values[m]),
                "x": xs[m],
                "ineqlin": duals[m],
                "eqlin": np.zeros(1),
                "fathomed": False,
                "conds": [],
                "node": node,
                "bounds": node["bounds"],
                "status": "assortment",
            }
            for m in order
        ]
