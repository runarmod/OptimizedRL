"""Evaluation of SRL policies on the DAP environment.

    uv run python -m src.srl.evaluate runs/srl/seed_14/best.pt --first-seed 2001
"""

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from src.gym_envs.dap_env import DynamicAssortmentEnvironment, DynamicAssortmentSimulator
from src.srl.checkpoint import load_agent
from src.srl.config import SrlEnvConfig
from src.utils.evaluation import run_episode

Policy = Callable[[np.ndarray], np.ndarray]


def make_env(config: SrlEnvConfig) -> DynamicAssortmentEnvironment:
    return DynamicAssortmentEnvironment(
        episode_seeds=range(1, config.n_train_seeds + 1),
        sim=DynamicAssortmentSimulator(
            n_items=config.n_items,
            n_static_features=2,
            assortment_size=config.assortment_size,
            initial_inventory=config.initial_inventory,
            max_steps=config.max_steps,
        ),
    )


def run_policy(
    env: DynamicAssortmentEnvironment, policy: Policy, seeds: Iterable[int], max_steps: int
) -> list[dict[str, float]]:
    """Episode totals of a policy that maps an observation to an assortment."""
    return [
        {"seed": seed, **run_episode(env, lambda: policy(env.state), seed, max_steps)}
        for seed in seeds
    ]


def mean_revenue(
    env: DynamicAssortmentEnvironment, policy: Policy, seeds: Iterable[int], max_steps: int
) -> float:
    return float(np.mean([row["ep_reward"] for row in run_policy(env, policy, seeds, max_steps)]))


def evaluate_srl(
    label: str, checkpoint: Path, seeds: list[int], max_steps: int | None = None
) -> list[dict[str, float]]:
    """Per-episode results of a saved SRL policy, as rows of `evaluate_params`."""
    agent, _ = load_agent(checkpoint)
    env_config = agent.config.env
    steps = max_steps or env_config.max_steps
    rows = run_policy(make_env(env_config), agent.best_assortment, seeds, steps)
    return [{"run": label, **row} for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRL checkpoints on DAP.")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    for checkpoint in args.checkpoints:
        rows = evaluate_srl(str(checkpoint), checkpoint, seeds)
        revenue = np.array([row["ep_reward"] for row in rows])
        print(f"{checkpoint}: mean revenue {revenue.mean():.1f} +/- {revenue.std():.1f}")


if __name__ == "__main__":
    main()
