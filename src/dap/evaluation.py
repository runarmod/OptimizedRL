from collections.abc import Callable, Sequence

import numpy as np

from src.dap.milp import assortment_values, initial_theta, model_features
from src.dap.simulator import DAPSimulator, all_assortments

Policy = Callable[[DAPSimulator], np.ndarray]

# Episode seeds of paper 02: train 1-100, validation 1001-1100, test 2001-2100.
TRAIN_SEEDS = range(1, 101)
VAL_SEEDS = range(1001, 1101)
TEST_SEEDS = range(2001, 2101)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(scores))
    out[np.argsort(-scores, kind="stable")[:k]] = 1.0
    return out


def greedy_policy(sim: DAPSimulator) -> np.ndarray:
    """Show the k most expensive items (paper 02's greedy baseline)."""
    return _top_k(sim.prices, sim.k)


def expert_policy(sim: DAPSimulator) -> np.ndarray:
    """Myopic expert: maximise expected revenue of this step with the true model.

    Enumerates all assortments, as paper 02's ``expert_solution``.
    """
    assortments = all_assortments(sim.n_items, sim.k)
    available = (assortments * (sim.inventory <= 0)).sum(axis=1) == 0
    weights = assortments * np.exp(sim.utilities())
    expected = (weights @ sim.prices) / (1.0 + weights.sum(axis=1))
    expected[~available] = -np.inf
    return assortments[int(np.argmax(expected))]


def milp_policy(theta: np.ndarray, n_pieces: int) -> Policy:
    """Deterministic CORL policy: the MILP optimum (argmin Q over assortments)."""
    theta = np.asarray(theta, dtype=float)
    cache = {}

    def policy(sim: DAPSimulator) -> np.ndarray:
        key = (sim.n_items, sim.k)
        if key not in cache:
            cache[key] = all_assortments(*key)
        assortments = cache[key]
        phi = model_features(sim.item_features())
        values, _ = assortment_values(theta, n_pieces, phi, assortments)
        return assortments[int(np.argmin(values))]

    return policy


def run_episode(sim: DAPSimulator, policy: Policy, seed: int) -> float:
    sim.reset(seed)
    total = 0.0
    while not sim.is_terminated():
        _, revenue = sim.step(policy(sim))
        total += revenue
    return total


def evaluate(policy: Policy, seeds: Sequence[int], sim: DAPSimulator | None = None) -> np.ndarray:
    sim = sim or DAPSimulator()
    return np.asarray([run_episode(sim, policy, s) for s in seeds])


def delta_greedy_pct(rewards, greedy_rewards) -> float:
    """Improvement over greedy in % of greedy's mean revenue."""
    return float((np.mean(rewards) - np.mean(greedy_rewards)) / np.mean(greedy_rewards) * 100)


def find_initial_theta(
    n_pieces: int,
    seed: int,
    reward_range: tuple[float, float] | None,
    check_seeds: Sequence[int] = range(1, 11),
    max_tries: int = 1000,
) -> tuple[np.ndarray, int, float]:
    """Random initial parameters, optionally restricted to a reward range.

    Paper 02 picks initial-model seeds whose policy earns 100-200 on episodes
    1-10, far below greedy, to avoid convergence by chance. This rejection
    samples Glorot initialisations until the deterministic MILP policy does
    the same. Returns (theta, tries, mean reward).
    """
    rng = np.random.default_rng(seed)
    sim = DAPSimulator()
    for tries in range(1, max_tries + 1):
        theta = initial_theta(n_pieces, rng)
        reward = float(evaluate(milp_policy(theta, n_pieces), check_seeds, sim).mean())
        if reward_range is None or reward_range[0] <= reward <= reward_range[1]:
            return theta, tries, reward
    raise RuntimeError(f"no initialisation with reward in {reward_range} in {max_tries} tries")


class DAPValidation:
    """Evaluation during training, following paper 02.

    The deterministic policy is evaluated on 30 train episodes (seeds 1-30)
    and 30 validation episodes (seeds 1001-1030); paper 02 keeps the model
    with the best mean of the two.
    """

    def __init__(self, n_pieces: int, n_episodes: int = 30, selection: str = "train_val_mean"):
        self.n_pieces = n_pieces
        self.seeds = {"train": TRAIN_SEEDS[:n_episodes], "val": VAL_SEEDS[:n_episodes]}
        self.selection = selection
        self.sim = DAPSimulator()
        self.greedy = {k: evaluate(greedy_policy, s, self.sim) for k, s in self.seeds.items()}
        self.best = -np.inf

    def evaluate(self, theta: np.ndarray) -> dict[str, float]:
        policy = milp_policy(theta, self.n_pieces)
        metrics = {}
        for name, seeds in self.seeds.items():
            rewards = evaluate(policy, seeds, self.sim)
            metrics[f"dap/{name}_reward"] = float(rewards.mean())
            metrics[f"dap/{name}_greedy_reward"] = float(self.greedy[name].mean())
            metrics[f"dap/{name}_delta_greedy_pct"] = delta_greedy_pct(rewards, self.greedy[name])
        if self.selection == "train_val_mean":
            metrics["dap/selection_reward"] = (metrics["dap/train_reward"] + metrics["dap/val_reward"]) / 2
        else:
            metrics["dap/selection_reward"] = metrics["dap/val_reward"]
        return metrics

    def is_new_best(self, metrics: dict[str, float]) -> bool:
        if metrics["dap/selection_reward"] > self.best:
            self.best = metrics["dap/selection_reward"]
            return True
        return False
