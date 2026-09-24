from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    aA: np.ndarray
    aB: np.ndarray
    b: np.ndarray
    c: np.ndarray


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

    def get_params(self) -> ModelParameters:
        return ModelParameters(
            aA=self.aA.copy(),
            aB=self.aB.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
        )

    @abstractmethod
    def update_params(self, grad, lr):
        pass

    @abstractmethod
    def get_desc_var_indices(self):
        pass
