import heapq

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from src.solvers.solver_interface import Solver


class BranchEnumerationSolver(Solver):
    """Candidate pool from best-first branching on binary expressions.

    When the LP relaxation of a MILP is integral (e.g. the network-flow
    structure of the DVSP), branch-and-bound stops at the root and exposes a
    single candidate, so the CORL softmax policy has nothing to sample from.
    This solver builds the tree explicitly instead: starting from the root
    optimum, a child fixes one more expression g(x) to the opposite of its
    current value (g(x) <= 0 or g(x) >= 1) and is solved as an LP. Children
    are explored best-first on their objective until ``max_pool_size``
    distinct solutions are found, up to ``max_depth`` fixings from the root.

    The expressions are taken from ``node["branch_exprs"]`` (one row per
    expression). Without them, every binary variable is its own expression.
    Branching rows do not depend on the model parameters, so their duals are
    dropped and the returned duals refer to the model's own constraints, as
    the envelope-theorem gradient requires.

    Every returned candidate is a B&B node with an exact LP value. Integral
    ones are leaves (``fathomed=False``). Fractional ones are dropped when
    ``integral_only`` is set, and otherwise returned as pruned nodes
    (``fathomed=True``) for the neighbourhood samplers.
    """

    def __init__(
        self,
        max_pool_size: int = 16,
        max_depth: int = 1,
        integral_only: bool = True,
        int_tol: float = 1e-6,
    ):
        self.max_pool_size = int(max_pool_size)
        self.max_depth = int(max_depth)
        self.integral_only = bool(integral_only)
        self.int_tol = float(int_tol)

    def solve(self, node):
        n_vars = len(node["c"])
        if n_vars == 0:
            return [self._entry(node, np.zeros(0), 0.0, None, None, (), True)]

        exprs = node.get("branch_exprs")
        if exprs is None:
            binary = [
                i
                for i, (is_int, bnd) in enumerate(zip(node["integer"], node["bounds"]))
                if is_int and tuple(bnd) == (0, 1)
            ]
            exprs = sparse.eye(n_vars, format="csr")[binary]
        exprs = sparse.csr_matrix(exprs)

        root = self._solve_lp(node, exprs, ())
        if root is None:
            return []

        pool = []
        seen = set()
        counter = 0
        frontier = []

        def add(entry) -> bool:
            sig = tuple(np.round(entry["x"], 6).tolist())
            if sig in seen:
                return False
            seen.add(sig)
            pool.append(entry)
            return True

        def expand(entry):
            nonlocal counter
            fixings = entry["fixings"]
            if len(fixings) >= self.max_depth:
                return
            fixed = {g for g, _ in fixings}
            values = exprs @ entry["x"]
            for g in range(exprs.shape[0]):
                if g in fixed:
                    continue
                target = 0 if values[g] >= 0.5 else 1
                child = self._solve_lp(node, exprs, fixings + ((g, target),))
                if child is not None:
                    counter += 1
                    heapq.heappush(frontier, (child["fun"], counter, child))

        add(root)
        expand(root)
        while frontier and len(pool) < self.max_pool_size:
            _, _, entry = heapq.heappop(frontier)
            if add(entry):
                expand(entry)
        return pool

    def _solve_lp(self, node, exprs, fixings):
        A_ub = node["A_ub"]
        b_ub = node["b_ub"]
        n_ub = 0 if A_ub is None else A_ub.shape[0]
        rows = []
        rhs = []
        for g, target in fixings:
            if target == 0:
                rows.append(exprs[g])  # g(x) <= 0
                rhs.append(0.0)
            else:
                rows.append(-exprs[g])  # g(x) >= 1
                rhs.append(-1.0)
        if rows:
            extra = sparse.vstack(rows)
            A_ub = extra if A_ub is None else sparse.vstack((sparse.csr_matrix(A_ub), extra))
            b_ub = np.concatenate((np.zeros(0) if b_ub is None else b_ub, rhs))

        res = linprog(
            node["c"],
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=node["A_eq"],
            b_eq=node["b_eq"],
            bounds=node["bounds"],
            method="highs-ds",
        )
        if not res.success:
            return None

        x = np.asarray(res.x, dtype=float)
        integral = self._is_integral(x, node["integer"])
        if not integral and self.integral_only:
            return None
        ineqlin = (
            np.asarray(res.ineqlin.marginals, dtype=float)[:n_ub]
            if n_ub
            else np.zeros(0)
        )
        eqlin = (
            np.asarray(res.eqlin.marginals, dtype=float)
            if node["A_eq"] is not None
            else np.zeros(0)
        )
        if integral:
            x = np.where(np.asarray(node["integer"], dtype=bool), np.round(x), x)
        return self._entry(node, x, float(res.fun), ineqlin, eqlin, fixings, integral, res)

    def _is_integral(self, x, integer) -> bool:
        mask = np.asarray(integer, dtype=bool)
        return bool(np.all(np.abs(x[mask] - np.round(x[mask])) <= self.int_tol))

    @staticmethod
    def _entry(node, x, fun, ineqlin, eqlin, fixings, integral, res=None):
        return {
            "fun": fun,
            "x": x,
            "ineqlin": np.zeros(0) if ineqlin is None else ineqlin,
            "eqlin": np.zeros(0) if eqlin is None else eqlin,
            "lower": None if res is None else np.asarray(res.lower.marginals),
            "upper": None if res is None else np.asarray(res.upper.marginals),
            "fathomed": not integral,
            "conds": [],
            "fixings": fixings,
            "node": node,
            "bounds": node["bounds"],
            "status": "root" if not fixings else "branch",
        }
