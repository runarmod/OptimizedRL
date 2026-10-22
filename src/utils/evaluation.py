from collections.abc import Callable

import numpy as np

from src.config.config_models import AppConfig
from src.gym_envs.env_interface import Env
from src.models.model_interface import Model
from src.solvers.solver_interface import Solver


def best_action(model: Model, solver: Solver) -> np.ndarray:
    """Most probable action of the trained policy."""
    return solver.solve_best(model.get_LP_formulation(), model)


def seed_episode(env: Env, seed: int) -> None:
    np.random.seed(seed)
    env.reset(seed=seed)


def run_episode(
    env: Env, act: Callable[[], np.ndarray], seed: int, max_steps: int
) -> dict[str, float]:
    seed_episode(env, seed)
    totals = {"ep_reward": 0.0}
    length = 0
    for _ in range(max_steps):
        _, reward, terminated, _, info = env.step(act())
        totals["ep_reward"] += float(reward)
        for name, value in env.get_plot_metrics(info).items():
            totals[name] = totals.get(name, 0.0) + value
        length += 1
        if terminated:
            break
    totals["ep_length"] = length
    return totals


def default_max_steps(
    requested: int | None, config: AppConfig, problem: str | None = None
) -> int:
    """Steps per episode: `requested` if given, else the default of `problem`."""
    if requested is not None:
        return requested
    if (problem or config.problem) == "dap":
        return config.dap_env.max_steps
    return getattr(config, config.training.algorithm).rollout_iters
