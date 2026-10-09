"""Evaluate trained model parameters with deterministic (best) actions.

Every parameter file is evaluated on the same seeded episodes, so the results
of different training runs can be compared directly.

    uv run python evaluate.py params/<run id>.yaml params/<other run id>.yaml 
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from src.config.config_loader import load_config
from src.config.config_models import AppConfig
from src.gym_envs.env_interface import Env
from src.models.model_interface import Model
from src.solvers.solver_interface import Solver
from src.utils.policy import nn_branch_sample
from train import build_model_and_env, build_solver, sanitize_env_action


def best_action(model: Model, solver: Solver) -> np.ndarray:
    """Most probable action of the trained policy.
    """
    node = model.get_LP_formulation()
    sol_pool = solver.solve(node)
    desc_vars = model.get_desc_var_indices()
    if not sol_pool:
        return np.zeros(len(node["c"]))[desc_vars]

    best = min(sol_pool, key=lambda sol: sol["fun"])
    action = best["x"][desc_vars]
    if best.get("fathomed"):
        action, _ = nn_branch_sample(action, best["bounds"])
    return action


def seed_episode(env: Env, seed: int) -> None:
    np.random.seed(seed)
    env.reset(seed=seed)


def run_episode(
    model: Model, env: Env, solver: Solver, seed: int, max_steps: int
) -> dict[str, float]:
    seed_episode(env, seed)
    totals = {"ep_reward": 0.0}
    length = 0
    for _ in range(max_steps):
        model.update_from_environment(env)
        action = sanitize_env_action(env, best_action(model, solver))
        _, reward, terminated, _, info = env.step(action)
        totals["ep_reward"] += float(reward)
        for name, value in env.get_plot_metrics(info).items():
            totals[name] = totals.get(name, 0.0) + value
        length += 1
        if terminated:
            break
    totals["ep_length"] = length
    return totals


def config_for_params(params_path: Path, fallback: AppConfig) -> AppConfig:
    """Rebuild the config the parameters were trained with, set to load them."""
    with params_path.open() as params_file:
        saved = yaml.safe_load(params_file)
    if "config" in saved:
        config = AppConfig.model_validate(saved["config"])
    else:
        print(f"{params_path} has no saved config, using config.yaml instead.")
        config = fallback
    return config.model_copy(update={"load": True, "load_path": params_path})


def evaluate(
    label: str, config: AppConfig, project_root: Path, seeds: list[int], max_steps: int
) -> list[dict[str, float]]:
    model, env = build_model_and_env(config, project_root)
    solver = build_solver(config)
    results = []
    for seed in seeds:
        episode = run_episode(model, env, solver, seed, max_steps)
        results.append({"run": label, "seed": seed, **episode})
    return results


def print_summary(results: list[dict[str, float]]) -> None:
    labels = list(dict.fromkeys(row["run"] for row in results))
    metric_names = [k for k in results[0] if k not in ("run", "seed")]
    label_width = max(len(label) for label in labels)
    header = f"{'run':<{label_width}}" + "".join(
        f"  {name:>26}" for name in metric_names
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        rows = [row for row in results if row["run"] == label]
        cells = []
        for name in metric_names:
            values = np.array([row[name] for row in rows], dtype=float)
            cells.append(f"{values.mean():>12.4f} +/- {values.std():<9.4f}")
        print(f"{label:<{label_width}}" + "".join(f"  {cell:>26}" for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", nargs="*", type=Path, help="Saved params files.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the first episode."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Step limit per episode (default: rollout_iters of the training run).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also evaluate the untrained initial parameters from config.yaml.",
    )
    parser.add_argument("--csv", type=Path, help="Write per-episode results here.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_config = load_config(project_root / "config.yaml")
    if not args.params and not args.baseline:
        parser.error("give at least one params file or --baseline")

    runs: list[tuple[str, AppConfig]] = []
    if args.baseline:
        runs.append(("baseline", base_config.model_copy(update={"load": False})))
    for params_path in args.params:
        runs.append(
            (params_path.stem, config_for_params(params_path.resolve(), base_config))
        )

    seeds = list(range(args.seed, args.seed + args.episodes))
    results = []
    for label, config in runs:
        max_steps = args.max_steps
        if max_steps is None:
            algorithm_config = getattr(config, config.training.algorithm)
            max_steps = algorithm_config.rollout_iters
        print(f"Evaluating {label} on {len(seeds)} episodes...")
        results.extend(evaluate(label, config, project_root, seeds, max_steps))

    print()
    print_summary(results)

    if args.csv:
        fieldnames = list(dict.fromkeys(k for row in results for k in row))
        with args.csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-episode results written to {args.csv}")


if __name__ == "__main__":
    main()
