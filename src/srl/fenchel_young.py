"""Perturbed Fenchel-Young loss.

Port of InferOpt's `FenchelYoungLoss(PerturbedAdditive(DAP_optimization; ε,
nb_samples))` as used in the SRL codebase. With perturbations eta_k = theta +
epsilon Z_k and the CO-layer f, the loss is

    F(theta) - <theta, target>,   F(theta) = mean_k <eta_k, f(eta_k)>,

with gradient mean_k f(eta_k) - target with respect to theta.
"""

import torch

from src.srl.co_layer import top_k


def perturbed_fenchel_young_loss(
    theta: torch.Tensor,
    target: torch.Tensor,
    k: int,
    epsilon: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Loss (batch,) of scores theta (batch, n_items) for target actions
    (batch, n_items), given standard normal noise (batch, n_samples, n_items)."""
    with torch.no_grad():
        eta = theta.unsqueeze(-2) + epsilon * noise
        actions = top_k(eta, k)
        f_value = (eta * actions).sum(-1).mean(-1)
        mean_action = actions.mean(-2)
    # F has gradient mean_action (Danskin's theorem), and the value F.
    linear = (theta * mean_action).sum(-1)
    f = linear + (f_value - linear).detach()
    return f - (theta * target.detach()).sum(-1)
