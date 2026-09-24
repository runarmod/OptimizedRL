from abc import ABC, abstractmethod


class Model(ABC):
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

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def update_params(self, grad, lr):
        pass

    @abstractmethod
    def get_desc_var_indices(self):
        pass
