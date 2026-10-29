import csv

import numpy as np
import pytest
import torch

from src.srl.agent import SrlAgent
from src.srl.checkpoint import build_critics, load_agent, read_checkpoint
from src.srl.config import SrlConfig
from src.srl.evaluate import make_env
from src.srl.networks import Actor
from src.srl.train import train, validate


def tiny_config(**updates) -> SrlConfig:
    config = SrlConfig.model_validate(
        {
            "episodes": 2,
            "iterations": 3,
            "future_critic_passes": 2,
            "validation": {"episodes": 2},
            "init": {"reward_range": None},
        }
    )
    return config.model_copy(update=updates)


def new_agent(config: SrlConfig) -> SrlAgent:
    generator = torch.Generator().manual_seed(config.seed)
    return SrlAgent(config, Actor(generator), *build_critics(config, generator))


def test_collected_episode_replays_in_the_environment():
    config = tiny_config()
    agent = new_agent(config)
    transitions = agent.collect_episode(make_env(config.env), 7)
    assert len(transitions) == config.env.max_steps

    env = make_env(config.env)
    observation, _ = env.reset(seed=7)
    for transition in transitions:
        np.testing.assert_allclose(
            transition.features.ravel(), observation, rtol=1e-6
        )
        assert transition.assortment.sum() == config.env.assortment_size
        observation, revenue, _, _, _ = env.step(transition.assortment.astype(np.int32))
        assert revenue == pytest.approx(transition.revenue)


def test_update_changes_the_actor_and_reports_losses():
    config = tiny_config()
    agent = new_agent(config)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    metrics = agent.update(agent.collect_episode(make_env(config.env), 1))
    assert set(metrics) == {"actor_loss", "reward_critic_loss", "future_critic_loss"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(p, q) for p, q in zip(before, agent.actor.parameters()))
    # Gradients are clipped elementwise before Adam, whose first steps move each
    # parameter by about the learning rate.
    for p, q in zip(before, agent.actor.parameters()):
        assert (p - q).abs().max() <= config.iterations * config.lr.start * 1.01


def test_schedules_advance_per_episode():
    config = tiny_config(episodes=4)
    agent = new_agent(config)
    assert agent.sigma_f == 2.0
    agent.end_episode()
    assert agent.sigma_f == pytest.approx(1.75)
    assert agent.actor_opt.param_groups[0]["lr"] == pytest.approx(1e-3 - 1.25e-4)


def read_history(path) -> list[dict]:
    with path.open() as history_file:
        rows = list(csv.DictReader(history_file))
    for row in rows:
        row.pop("elapsed_s")
    return rows


def test_training_is_deterministic_and_best_checkpoint_reloads(tmp_path):
    config = tiny_config()
    best = train(config, tmp_path / "a")
    train(config, tmp_path / "b")
    history = read_history(tmp_path / "a" / "history.csv")
    assert history == read_history(tmp_path / "b" / "history.csv")
    assert len(history) == config.episodes

    saved = read_checkpoint(best)
    assert saved["train_time_s"] > 0
    agent, _ = load_agent(best)
    rewards = validate(agent, make_env(config.env), config)
    best_row = history[saved["episode"] - 1]
    assert float(best_row["score"]) == pytest.approx(np.mean(list(rewards.values())))
