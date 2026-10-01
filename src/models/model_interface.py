from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ParameterSnapshot(ABC):
    @abstractmethod
    def change_metrics(self, original: "ParameterSnapshot") -> dict[str, float]:
        """Squared change of each parameter group relative to ``original``."""


@dataclass(frozen=True)
class ModelParameters(ParameterSnapshot):
    aA: np.ndarray
    aB: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def change_metrics(self, original: "ModelParameters") -> dict[str, float]:
        return {
            "c_change": ((-original.c - self.c) ** 2).sum(),
            "aA_change": ((original.aA - self.aA) ** 2).sum(),
            "aB_change": ((original.aB - self.aB) ** 2).sum(),
            "b_change": ((original.b - self.b) ** 2).sum(),
        }


@dataclass(frozen=True)
class VectorModelParameters(ParameterSnapshot):
    """Parameters of models whose learnable part is a single flat vector."""

    name: str
    values: np.ndarray

    def change_metrics(self, original: "VectorModelParameters") -> dict[str, float]:
        metrics = {f"{self.name}_change": ((original.values - self.values) ** 2).sum()}
        for i, value in enumerate(self.values):
            metrics[f"{self.name}_{i}"] = float(value)
        return metrics


class Model(ABC):
    def __init__(self):
        self.s_t = None
        self.aA = None
        self.aB = None
        self.b = None
        self.c = None

    def _update_state(self, state):
        self.s_t = state

    @abstractmethod
    def update_from_environment(self, environment):
        pass

    @abstractmethod
    def get_LP_formulation(self):
        pass

    @abstractmethod
    def lagrange_gradient(self):
        pass

    def get_params(self) -> ParameterSnapshot:
        return ModelParameters(
            aA=self.aA.copy(),
            aB=self.aB.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
        )

    def get_policy_params(self) -> np.ndarray:
        """Flat vector of the parameters trained by PPO."""
        return np.concatenate(
            [
                np.asarray(self.aA, dtype=np.float32).reshape(-1),
                np.asarray(self.aB, dtype=np.float32).reshape(-1),
                np.asarray(self.b, dtype=np.float32).reshape(-1),
            ]
        )

    def set_policy_params(self, theta: np.ndarray) -> None:
        """Inverse of ``get_policy_params``."""
        idx = 0
        for name in ("aA", "aB", "b"):
            shape = getattr(self, name).shape
            size = int(np.prod(shape))
            setattr(self, name, theta[idx : idx + size].reshape(shape).astype(float))
            idx += size

    @abstractmethod
    def update_params(self, grad, lr):
        pass

    @abstractmethod
    def get_desc_var_indices(self):
        pass
