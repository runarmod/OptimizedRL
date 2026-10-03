import gymnasium as gym
import numpy as np


class DynamicAssortmentEnvironment(gym.Env):
    def __init__(
        self,
        N: int,
        d: int,
        K: int,
        J: int,
        max_steps: int = 80,
        seed: int = 0,
        dynamic: bool = True,
    ):
        self.N = N  # Number of items
        self.d = d  # Dimension of feature vectors
        self.K = K  # Assortment size constraint
        self.J = J  # Initial inventory
        self.max_steps = max_steps  # Steps per history
        self.dynamic = dynamic  # Endogeneity of the customer model
        self.purchase_hist: list[int] = []
        self.step = 0
        self.seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            super().reset(seed=seed)

        rng = np.random.default_rng(self.seed)
        self.features = rng.uniform(1.0, 10.0, (self.d + 3, self.N))
        self.features = np.vstack((self.features, np.ones((1, self.N))))
        self.d_features = np.zeros((2, self.N))
        self.inventory = np.ones(self.N)
        self.step = 1


if __name__ == "__main__":
    dap = DynamicAssortmentEnvironment(20, 2, 3, 200)
    dap.reset()
    print(dap.features)
