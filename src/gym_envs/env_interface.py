from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def get_plot_metrics(self, info: dict) -> dict[str, float]:
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, seed):
        pass
