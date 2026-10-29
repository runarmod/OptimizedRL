"""Combinatorial optimization layer of the SRL pipeline for DAP."""

import numpy as np
import torch


def top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Assortments of the k highest scores along the last axis, as 0/1 floats.

    `DAP_optimization` in the SRL codebase: argmax of <scores, a> over the
    assortments a showing exactly k items.
    """
    shown = scores.topk(k, dim=-1).indices
    return torch.zeros_like(scores).scatter_(-1, shown, 1.0)


def top_k_numpy(scores: np.ndarray, k: int) -> np.ndarray:
    """`top_k` of one score vector, as the int assortment the environment takes."""
    assortment = np.zeros(scores.shape[-1], dtype=np.int32)
    assortment[np.argpartition(-scores, k - 1)[:k]] = 1
    return assortment
