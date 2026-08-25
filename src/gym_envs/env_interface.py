from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, seed):
        pass
