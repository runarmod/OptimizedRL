from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from src.gym_envs.env_interface import Env

# Rows of the feature matrix, counted from the end so they hold for any number
# of static features (which come first).
HYPE, SATISFACTION, PRICE, TIME = -4, -3, -2, -1

# Hidden customer model - assumes 2 static features
CUSTOMER_MODEL = np.array([0.3, 0.5, 0.6, -0.4, -0.8, 0.0])

# Per-item observation: the feature rows, the one-step change of hype and
# satisfaction, and their change since the start of the episode.
N_ITEM_FEATURES = len(CUSTOMER_MODEL) + 2 + 2


class DynamicAssortmentSimulator:
    def __init__(
        self,
        n_items: int,
        n_static_features: int,
        assortment_size: int,
        initial_inventory: int,
        max_steps: int = 80,
        seed: int = 0,
    ) -> None:
        # Number of items to choose from (N in SRL codebase)
        self.n_items = n_items

        # Dimension of feature vectors (in addition to hype, satisfaction, and price) (d in SRL codebase)
        assert n_static_features == 2, "CUSTOMER_MODEL needs 2 static features"
        self.n_static_features = n_static_features

        # Assortment size constraint, i.e. how many items to show (K in SRL codebase)
        self.assortment_size = assortment_size

        # Initial inventory of each item (J in SRL codebase)
        self.initial_inventory = initial_inventory
        self.max_steps = max_steps  # Steps per history
        self.purchase_hist: list[int] = []
        self.current_step = 0
        self.seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed

        rng = np.random.default_rng(self.seed)
        self.features = rng.uniform(
            1.0, 10.0, (self.n_static_features + 3, self.n_items)
        )
        self.features = np.vstack((self.features, np.ones((1, self.n_items))))
        self.start_features = self.features.copy()
        self.d_features = np.zeros((2, self.n_items))
        self.inventory = np.full(self.n_items, self.initial_inventory)
        self.purchase_hist = []
        self.current_step = 1

    def hype_update(self) -> np.ndarray:
        hype_vector = np.ones(self.n_items)

        latest_item = self.purchase_hist[-1]
        if latest_item != -1:
            hype_vector[latest_item] += 0.02

        for offset in range(2, 6):
            if len(self.purchase_hist) < offset:
                break

            previous_item = self.purchase_hist[-offset]
            if previous_item != -1:
                hype_vector[previous_item] -= 0.005

        return hype_vector

    def apply_purchase(self, item: int) -> None:
        old_features = np.copy(self.features)
        self.purchase_hist.append(item)

        hype_vector = self.hype_update()
        self.features[HYPE] *= hype_vector
        if item != -1:
            self.features[SATISFACTION, item] *= 1.01
        self.features[TIME] += 9 / self.max_steps

        self.d_features = self.features[HYPE:PRICE] - old_features[HYPE:PRICE]
        if item != -1:
            self.inventory[item] -= 1

        self.current_step += 1

    def item_features(self) -> np.ndarray:
        """Shape (n_items, N_ITEM_FEATURES), `feature_vector` in the SRL codebase."""
        delta = self.features[HYPE:PRICE] - self.start_features[HYPE:PRICE]
        return np.vstack((self.features, self.d_features, delta)).T

    def customer_utilities(self) -> np.ndarray:
        return CUSTOMER_MODEL @ self.features

    @staticmethod
    def choice_probabilities(utility: np.ndarray, assortment: np.ndarray) -> np.ndarray:
        weight = np.exp(utility) * assortment
        return np.append(weight, 1) / (1 + weight.sum())

    @property
    def prices(self) -> np.ndarray:
        return self.features[PRICE]

    def price(self, item: int) -> float:
        return float(self.prices[item])

    def purchase(self, assortment: np.ndarray) -> tuple[int, float]:
        if (
            assortment.shape != (self.n_items,)
            or np.count_nonzero(assortment) != self.assortment_size
        ):
            raise ValueError(
                f"assortment must show exactly {self.assortment_size}"
                f" of {self.n_items} items"
            )
        probabilities = self.choice_probabilities(self.customer_utilities(), assortment)
        rng = np.random.default_rng(self.seed + self.current_step)
        choice_idx = int(rng.choice(self.n_items + 1, p=probabilities))
        if choice_idx == self.n_items:
            return -1, 0.0
        return choice_idx, self.price(choice_idx)

    def is_terminated(self) -> bool:
        return (
            self.current_step > self.max_steps
            or int(np.count_nonzero(self.inventory > 0)) < self.assortment_size
        )

    def step(self, assortment: np.ndarray) -> tuple[int, float]:
        item, revenue = self.purchase(assortment)
        self.apply_purchase(item)
        return item, revenue


class DynamicAssortmentEnvironment(Env):
    def __init__(
        self,
        episode_seeds: Sequence[int],
        sim: DynamicAssortmentSimulator | None = None,
    ) -> None:
        self.sim = sim or DynamicAssortmentSimulator(
            n_items=20,
            n_static_features=2,
            assortment_size=4,
            initial_inventory=200,
        )
        n_items = self.sim.n_items
        self.action_space = gym.spaces.MultiDiscrete(np.full(n_items, 2))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (n_items * N_ITEM_FEATURES,), np.float32
        )
        self.episode_seeds = episode_seeds
        self.episode = 0
        self.state: np.ndarray | None = None

    def _observation(self) -> np.ndarray:
        return self.sim.item_features().astype(np.float32).ravel()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self.episode_seeds[self.episode % len(self.episode_seeds)]
        self.episode += 1
        self.sim.reset(seed)
        self.state = self._observation()
        return self.state, {"episode_seed": seed}

    def step(self, action):
        old_state = self.state
        # Mean revenue of this assortment under the customer model: a less noisy
        # learning curve than the reward, which is one sampled purchase.
        probabilities = self.sim.choice_probabilities(
            self.sim.customer_utilities(), action
        )
        expected_revenue = float(probabilities[:-1] @ self.sim.prices)
        item, revenue = self.sim.step(action)
        self.state = self._observation()
        info = {
            "action": item,
            "old_state": old_state,
            "new_state": self.state,
            "revenue": revenue,
            "expected_revenue": expected_revenue,
        }
        return self.state, revenue, self.sim.is_terminated(), False, info

    def get_plot_metrics(self, info: dict) -> dict[str, float]:
        return {
            "expected_revenue": info["expected_revenue"],
            "purchased": float(info["action"] != -1),
        }
