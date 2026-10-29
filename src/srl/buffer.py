"""Experience of the SRL agent: transitions, returns and the replay buffer."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Transition:
    features: np.ndarray  # (n_items, N_ITEM_FEATURES), `feat_t`
    assortment: np.ndarray  # (n_items,) executed 0/1 assortment, `S_t`
    revenue: float  # `rev_t`
    ret: float = 0.0  # discounted return from this step, `ret_t`


def discounted_returns(revenues: np.ndarray, gamma: float) -> np.ndarray:
    """Returns as `generate_episode` in the SRL codebase computes them.

    The recursion starts at the second-to-last step with only its own revenue,
    and the last step keeps a return of 0. The future critic learns from these
    values, so the port keeps this behaviour.
    """
    n = len(revenues)
    returns = np.zeros(n)
    for i in range(n - 2, -1, -1):
        returns[i] = revenues[i] if i == n - 2 else revenues[i] + gamma * returns[i + 1]
    return returns


def with_returns(transitions: list[Transition], gamma: float) -> list[Transition]:
    revenues = np.array([t.revenue for t in transitions])
    returns = discounted_returns(revenues, gamma)
    return [replace(t, ret=float(r)) for t, r in zip(transitions, returns)]


class ReplayBuffer:
    """Ring buffer sampled uniformly with replacement (`rb_add`, `rb_sample`)."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.items: list[Transition] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            if len(self.items) < self.capacity:
                self.items.append(transition)
            else:
                self.items[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        indices = self.rng.integers(len(self.items), size=batch_size)
        return [self.items[i] for i in indices]
