"""Actor and critics of the SRL pipeline for DAP.

Port of the models built in scripts/DAP/00_setup.jl and used by `critic_reward`
and `critic_future` in scripts/DAP/utils/utils.jl of the SRL codebase. Flux
applies a Dense layer to every column of a feature matrix; here the items are
the rows, so every per-item layer runs on the last axis. No layer has an
activation, as in the SRL codebase.
"""

import numpy as np
import torch
from torch import nn

from src.gym_envs.dap_env import N_ITEM_FEATURES

# The reward critic sees the first feature rows of the shown items: the static
# features, hype, satisfaction, price and time (`features[1:6, :]`).
N_REWARD_FEATURES = 6

ACTOR_HIDDEN = 5
REWARD_CRITIC_HIDDEN = 3
FUTURE_CRITIC_ITEM_HIDDEN = 5
FUTURE_CRITIC_HIDDEN = 10


def dense(n_in: int, n_out: int, generator: torch.Generator) -> nn.Linear:
    """Linear layer with Flux's default initialization: Glorot-uniform weights
    and zero biases."""
    layer = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform_(layer.weight, generator=generator)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Statistical model phi_w: one score per item from its features.

    `Chain(Dense(10 => 5), Dense(5 => 1), vec)` in the SRL codebase.
    """

    def __init__(self, generator: torch.Generator):
        super().__init__()
        self.hidden = dense(N_ITEM_FEATURES, ACTOR_HIDDEN, generator)
        self.out = dense(ACTOR_HIDDEN, 1, generator)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Scores (..., n_items) from features (..., n_items, N_ITEM_FEATURES)."""
        return self.out(self.hidden(features)).squeeze(-1)

    def linear_scores(self) -> tuple[np.ndarray, float]:
        """Weights w and bias b with scores = features @ w + b.

        Both layers are linear, so the actor is one linear map. Evaluating it
        with numpy is much faster than a torch forward pass per step.
        """
        with torch.no_grad():
            w1, b1 = self.hidden.weight.double(), self.hidden.bias.double()
            w2, b2 = self.out.weight.double(), self.out.bias.double()
            w = (w2 @ w1).reshape(-1)
            b = (w2 @ b1 + b2).item()
        return w.numpy(), b


class RewardCritic(nn.Module):
    """Estimates the immediate expected revenue of an assortment.

    `Chain(Dense(6 => 3), vec, Dense(12 => 1), vec)` applied to the shown items
    in index order (`critic_reward` in the SRL codebase).
    """

    def __init__(self, assortment_size: int, generator: torch.Generator):
        super().__init__()
        self.assortment_size = assortment_size
        self.item = dense(N_REWARD_FEATURES, REWARD_CRITIC_HIDDEN, generator)
        self.out = dense(REWARD_CRITIC_HIDDEN * assortment_size, 1, generator)

    def forward(self, features: torch.Tensor, assortments: torch.Tensor) -> torch.Tensor:
        """Values (batch,) of binary assortments (batch, n_items) in states with
        features (batch, n_items, N_ITEM_FEATURES)."""
        batch = assortments.shape[0]
        # nonzero lists the shown items row by row in increasing index order, as
        # `features[:, S .!= 0]` does.
        shown = torch.nonzero(assortments > 0.5)[:, 1]
        if shown.numel() != batch * self.assortment_size:
            raise ValueError(
                f"every assortment must show exactly {self.assortment_size} items"
            )
        shown = shown.reshape(batch, self.assortment_size)
        rows = torch.arange(batch, device=features.device)[:, None]
        shown_features = features[rows, shown, :N_REWARD_FEATURES]
        hidden = self.item(shown_features).flatten(start_dim=1)
        return self.out(hidden).squeeze(-1)


class FutureCritic(nn.Module):
    """Estimates the discounted revenue after the current step.

    `Chain(Dense(11 => 5), vec, Dense(100 => 10), Dense(10 => 1), vec)` applied to
    the features of all items with the assortment indicator appended
    (`critic_future` in the SRL codebase).
    """

    def __init__(self, n_items: int, generator: torch.Generator):
        super().__init__()
        self.item = dense(N_ITEM_FEATURES + 1, FUTURE_CRITIC_ITEM_HIDDEN, generator)
        self.hidden = dense(
            FUTURE_CRITIC_ITEM_HIDDEN * n_items, FUTURE_CRITIC_HIDDEN, generator
        )
        self.out = dense(FUTURE_CRITIC_HIDDEN, 1, generator)

    def forward(self, features: torch.Tensor, assortments: torch.Tensor) -> torch.Tensor:
        """Values (batch,) of assortments (batch, n_items) in states with features
        (batch, n_items, N_ITEM_FEATURES)."""
        item_input = torch.cat((features, assortments.unsqueeze(-1)), dim=-1)
        hidden = self.item(item_input).flatten(start_dim=1)
        return self.out(self.hidden(hidden)).squeeze(-1)
