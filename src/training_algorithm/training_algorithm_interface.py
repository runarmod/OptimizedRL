from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AlgorithmTransition:
    old_state_for_buffer: np.ndarray
    reward: float
    terminated: bool
    action_number: int
    old_state: np.ndarray
    new_state: np.ndarray
    chosen_action_raw: np.ndarray | None
    executed_action: np.ndarray


@dataclass
class AlgorithmDecision:
    action: np.ndarray
    info: dict
    store: bool
    raw_action: np.ndarray | None = None


class TrainingAlgorithm(ABC):
    """Interface for algorithms that interact with the training loop."""

    rollout_iters: int

    @abstractmethod
    def act(self, state: np.ndarray) -> AlgorithmDecision:
        """Select an action and return metadata needed for learning."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, transition: AlgorithmTransition, info: dict) -> None:
        """Record one environment transition for a later training update."""
        raise NotImplementedError

    @abstractmethod
    def update(self, state: np.ndarray) -> dict:
        """Train from collected transitions and return update metrics."""
        raise NotImplementedError
