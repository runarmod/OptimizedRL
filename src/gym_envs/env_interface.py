from abc import ABC, abstractmethod

import gymnasium as gym


class Env(ABC, gym.Env):
    @abstractmethod
    def get_plot_metrics(self, info: dict) -> dict[str, float]:
        pass
