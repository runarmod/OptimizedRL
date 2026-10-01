from collections.abc import Sequence

import numpy as np

from src.dap.simulator import N_ITEM_FEATURES, DAPSimulator, choice_probabilities
from src.gym_envs.env_interface import Env


class DAPEnv(Env):
    """Training environment for the dynamic assortment problem.

    Episodes cycle through ``episode_seeds`` (paper 02 trains on the episode
    seeds 1-100) or draw them at random. The action is a 0/1 vector over the
    items with exactly k ones; the reward is the revenue of the step.
    ``self.state`` is a permutation-invariant summary for the value network
    (episode progress and the mean, std and max of each item feature); the
    model reads the per-item features from ``item_features()``.
    """

    def __init__(
        self,
        episode_seeds: Sequence[int],
        seed_order: str = "cycle",
        seed: int = 0,
        sim: DAPSimulator | None = None,
    ):
        if seed_order not in ("cycle", "random"):
            raise ValueError("seed_order must be 'cycle' or 'random'")
        self.sim = sim or DAPSimulator()
        self.episode_seeds = list(episode_seeds)
        self.seed_order = seed_order
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.state = None

    @property
    def state_dim(self) -> int:
        return 1 + 3 * N_ITEM_FEATURES

    def item_features(self) -> np.ndarray:
        return self.sim.item_features()

    def _summary(self) -> np.ndarray:
        f = self.sim.item_features()
        progress = (self.sim.step_count - 1) / self.sim.max_steps
        return np.concatenate(([progress], f.mean(0), f.std(0), f.max(0))).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.seed_order == "cycle":
            episode_seed = self.episode_seeds[self.episode % len(self.episode_seeds)]
        else:
            episode_seed = int(self.rng.choice(self.episode_seeds))
        self.episode += 1
        self.sim.reset(episode_seed)
        self.state = self._summary()
        return self.state, {"episode_seed": episode_seed}

    def sanitize_action(self, action):
        return np.rint(np.clip(np.asarray(action, dtype=float).reshape(-1), 0, 1))

    def step(self, action):
        sim = self.sim
        x = self.sanitize_action(action)[: sim.n_items]
        if x.shape[0] != sim.n_items or int(x.sum()) != sim.k:
            raise ValueError(f"assortment must show exactly {sim.k} of {sim.n_items} items")
        probs = choice_probabilities(sim.utilities(), x)
        expected_revenue = float(probs[:-1] @ sim.prices)
        shown_price = float(sim.prices[x > 0.5].mean())
        old_state = self.state
        item, revenue = sim.step(x)
        terminated = sim.is_terminated()
        self.state = self._summary()
        info = {
            "action": item,
            "old_state": old_state,
            "new_state": self.state,
            "revenue": revenue,
            "expected_revenue": expected_revenue,
            "purchased": float(item >= 0),
            "shown_price": shown_price,
        }
        return self.state, revenue, terminated, False, info

    def get_plot_metrics(self, info: dict) -> dict[str, float]:
        return {
            "revenue": float(info["revenue"]),
            "expected_revenue": float(info["expected_revenue"]),
            "purchased": float(info["purchased"]),
            "shown_price": float(info["shown_price"]),
        }
