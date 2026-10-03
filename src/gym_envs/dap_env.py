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
        self.current_step = 0
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
        self.current_step = 1

    def hype_update(self):
        hype_vector = np.ones(self.N)

        latest_item = self.purchase_hist[-1]
        if latest_item != 0:
            hype_vector[latest_item - 1] += 0.02

        for offset in range(2, 6):
            if len(self.purchase_hist) < offset:
                break

            previous_item = self.purchase_hist[-offset]
            if previous_item != 0:
                hype_vector[previous_item - 1] -= 0.005

        return hype_vector

    def step(self, features: np.ndarray, inventory: np.ndarray, item: int):
        old_features = np.copy(features)
        self.purchase_hist.append(item)

        if self.dynamic:
            hype_vector = self.hype_update()
            features[2, :] *= hype_vector
            if item != 0:
                features[3, item - 1] *= 1.01
            features[5, :] += 9 / self.max_steps

        d_features = features[2:4, :] - old_features[2:4, :]
        if item != 0:
            inventory[item - 1] -= 1 / self.J

        inventory = np.round(inventory, decimals=4)
        self.current_step += 1
        return features, d_features, inventory


if __name__ == "__main__":
    dap = DynamicAssortmentEnvironment(20, 2, 3, 200)
    dap.reset()
    print(dap.features)
    dap.step(dap.features, dap.inventory, 20)
    print(dap.features)
