"""Saving and loading of SRL networks with the config that trained them."""

import os
from pathlib import Path

import torch

from src.srl.agent import SrlAgent
from src.srl.config import SrlConfig
from src.srl.networks import Actor, FutureCritic, RewardCritic


def build_critics(
    config: SrlConfig, generator: torch.Generator
) -> tuple[RewardCritic, FutureCritic]:
    """Both critics, initialized from the generator."""
    return (
        RewardCritic(config.env.assortment_size, generator),
        FutureCritic(config.env.n_items, generator),
    )


def save_checkpoint(path: Path, agent: SrlAgent, **info) -> None:
    """Write the networks, the config and `info` (e.g. the episode). The file is
    renamed into place, so an interrupt never leaves a partial checkpoint."""
    data = {
        "config": agent.config.model_dump(mode="json"),
        "actor": agent.actor.state_dict(),
        "reward_critic": agent.reward_critic.state_dict(),
        "future_critic": agent.future_critic.state_dict(),
        **info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def read_checkpoint(path: Path) -> dict:
    return torch.load(path, weights_only=True)


def load_agent(path: Path) -> tuple[SrlAgent, dict]:
    """Agent with the saved networks, and the whole checkpoint."""
    data = read_checkpoint(path)
    config = SrlConfig.model_validate(data["config"])
    generator = torch.Generator().manual_seed(0)
    actor = Actor(generator)
    reward_critic, future_critic = build_critics(config, generator)
    actor.load_state_dict(data["actor"])
    reward_critic.load_state_dict(data["reward_critic"])
    future_critic.load_state_dict(data["future_critic"])
    return SrlAgent(config, actor, reward_critic, future_critic), data
