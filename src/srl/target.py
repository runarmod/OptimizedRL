"""Target action of the SRL actor update."""

import torch


def softmax_target(
    candidates: torch.Tensor, values: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Softmax-weighted mean of candidate actions.

    candidates (batch, n_candidates, n_items) are scored by the critic values
    (batch, n_candidates). Rows whose weights are not finite fall back to the
    best candidate, as in the SRL codebase.
    """
    probs = torch.softmax(values / temperature, dim=-1)
    target = (probs.unsqueeze(-1) * candidates).sum(-2)
    invalid = torch.isnan(target).any(-1)
    if invalid.any():
        rows = invalid.nonzero().squeeze(-1)
        target[rows] = candidates[rows, values[rows].argmax(-1)]
    return target
