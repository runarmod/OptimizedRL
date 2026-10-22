from abc import ABC, abstractmethod

import numpy as np

from src.models.model_interface import Model
from src.utils.policy import nn_branch_sample


class NoSolutionError(RuntimeError):
    """The solver found no feasible solution of the MILP."""


class Solver(ABC):
    @abstractmethod
    def solve(self, node):
        pass

    @abstractmethod
    def solve_best(self, node, model: Model) -> np.ndarray:
        """Action of the best solution of the MILP `node` built by `model`.

        Raises NoSolutionError if the solver finds no solution."""


def best_action_from_pool(solver: Solver, node, model: Model) -> np.ndarray:
    """`solve_best` of solvers without one of their own. Used for legacy solvers."""
    sol_pool = solver.solve(node)
    if not sol_pool:
        raise NoSolutionError("the solver returned no candidates")
    sol_pool = model.complete_candidates(sol_pool)

    best = min(sol_pool, key=lambda sol: sol["fun"])
    action = best["x"][model.get_desc_var_indices()]
    if best.get("fathomed"):
        action, _ = nn_branch_sample(action, best["bounds"])
    return action
