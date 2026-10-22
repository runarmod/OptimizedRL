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

    def solve_best(self, node, model: Model) -> np.ndarray:
        """Action of the best solution of the MILP `node` built by `model`.

        Raises NoSolutionError if the solver finds no solution."""

        sol_pool = self.solve(node)
        if not sol_pool:
            raise NoSolutionError("the solver returned no candidates")
        sol_pool = model.complete_candidates(sol_pool)

        best = min(sol_pool, key=lambda sol: sol["fun"])
        action = best["x"][model.get_desc_var_indices()]
        if best.get("fathomed"):
            action, _ = nn_branch_sample(action, best["bounds"])
        return action
