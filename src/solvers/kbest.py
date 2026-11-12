"""Candidate pool of the K best integer solutions of the MILP.

Option A of the thesis plan: instead of the nodes of a branch-and-bound tree,
the policy chooses between the K best feasible solutions of the MILP. They are
found by solving the MILP, cutting off its solution with a no-good cut, and
solving again, K times.
"""

import numpy as np
from pyscipopt import SCIP_PARAMSETTING, Model, quicksum
from scipy.optimize import linprog as _linprog

from src.solvers.scip_brute import _normalize_linprog_duals
from src.solvers.solver_interface import NoSolutionError, Solver

BINARY_TOL = 1e-9


class KBestSolver(Solver):
    """Returns the K best integer solutions of the MILP as the candidate pool.

    Each candidate is the LP of the MILP with its integer variables fixed to the
    solution, so it carries that LP's duals, as the other solvers' candidates do.
    No-good cuts need binary integer variables, so a MILP with general integer
    variables is refused.
    """

    def __init__(self, k: int = 8, verbose: bool = False):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.verbose = verbose

    @staticmethod
    def _integer_indices(node) -> np.ndarray:
        """Indices of the integer variables, which must all be binary."""
        int_idx = np.flatnonzero(np.asarray(node["integer"], dtype=bool))
        for idx in int_idx:
            lb, ub = node["bounds"][idx]
            if lb is None or ub is None or lb < -BINARY_TOL or ub > 1 + BINARY_TOL:
                raise ValueError(
                    "KBestSolver needs binary integer variables for its no-good "
                    f"cuts, but variable {idx} has bounds ({lb}, {ub})."
                )
        return int_idx

    def _build_model(self, node) -> tuple[Model, list]:
        """SCIP model of the MILP `node`."""
        model = Model("KBest")
        model.hideOutput(not self.verbose)
        # The DAP MILPs are small: cutting planes and the default heuristics
        # cost more than they save (about 20 ms instead of 50-1800 ms per solve
        # with the McCormick pair terms).
        model.setSeparating(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.FAST)

        c = node["c"]
        vars_ = []
        for idx, (lb, ub) in enumerate(node["bounds"]):
            vars_.append(
                model.addVar(
                    name=f"x_{idx}",
                    lb=float(lb) if lb is not None else None,
                    ub=float(ub) if ub is not None else None,
                    vtype="B" if node["integer"][idx] else "C",
                )
            )
        model.setObjective(
            quicksum(float(c[i]) * vars_[i] for i in np.flatnonzero(c)),
            sense="minimize",
        )
        for a, b, sense in (
            (node.get("A_ub"), node.get("b_ub"), "<="),
            (node.get("A_eq"), node.get("b_eq"), "=="),
        ):
            if a is None or b is None:
                continue
            for row, rhs in zip(a, b):
                expr = quicksum(float(row[i]) * vars_[i] for i in np.flatnonzero(row))
                if sense == "<=":
                    model.addCons(expr <= float(rhs))
                else:
                    model.addCons(expr == float(rhs))
        return model, vars_

    def _fixed_integer_lp(self, node, int_idx: np.ndarray, x_int: np.ndarray):
        """Candidate made from the LP of `node` with the integer variables fixed
        to x_int, or None if that LP has no solution."""
        bounds = list(node["bounds"])
        for idx, value in zip(int_idx, x_int):
            bounds[idx] = (float(value), float(value))

        res = _linprog(
            node["c"],
            A_ub=node.get("A_ub"),
            b_ub=node.get("b_ub"),
            A_eq=node.get("A_eq"),
            b_eq=node.get("b_eq"),
            bounds=bounds,
        )
        if not res.success:
            return None

        x = np.asarray(res.x, dtype=float)
        x[int_idx] = x_int
        return {
            "fun": float(res.fun),
            "x": x,
            "eqlin": _normalize_linprog_duals(res.eqlin.marginals, node.get("A_eq")),
            "ineqlin": _normalize_linprog_duals(
                res.ineqlin.marginals, node.get("A_ub")
            ),
            "lower": np.asarray(res.lower.marginals, dtype=float),
            "upper": np.asarray(res.upper.marginals, dtype=float),
            "fathomed": False,
            "conds": [],
            "node": node,
            "bounds": bounds,
            "status": "kbest",
        }

    def _solve_k(self, node, k: int) -> list[dict]:
        """Up to k best integer solutions of `node`, best first."""
        int_idx = self._integer_indices(node)
        model, vars_ = self._build_model(node)
        try:
            results = []
            for rank in range(k):
                model.optimize()
                if model.getNSols() == 0:
                    break
                sol = model.getBestSol()
                x_int = np.round([model.getSolVal(sol, vars_[i]) for i in int_idx])

                entry = self._fixed_integer_lp(node, int_idx, x_int)
                if entry is not None:
                    entry["rank"] = rank
                    results.append(entry)

                if rank == k - 1:
                    break
                # No-good cut: every later solution differs from x_int in at
                # least one integer variable.
                model.freeTransform()
                model.addCons(
                    quicksum(
                        1 - vars_[i] if value > 0.5 else vars_[i]
                        for i, value in zip(int_idx, x_int)
                    )
                    >= 1
                )

            if self.verbose:
                print(f"KBestSolver: {len(results)} of {k} solutions")
            return results
        finally:
            model.freeProb()

    def solve(self, node):
        return self._solve_k(node, self.k) or None

    def solve_best(self, node, model) -> np.ndarray:
        """Action of the optimal solution. Solves the MILP once."""
        results = self._solve_k(node, 1)
        if not results:
            raise NoSolutionError("SCIP found no solution of the MILP")
        return results[0]["x"][model.get_desc_var_indices()]
