from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from src.dvsp.instance import (
    VSPInstance,
    list_instance_files,
    read_vsp_instance,
    split_instances,
)
from src.dvsp.simulator import DynamicVSP
from src.dvsp.solvers import (
    GREEDY_PRIZE,
    LAZY_PRIZE,
    anticipative_solve,
    solve_prize_collecting,
)

Policy = Callable[[DynamicVSP], list[list[int]]]


def greedy_policy(sim: DynamicVSP) -> list[list[int]]:
    """Dispatch every request as soon as it appears."""
    prizes = np.full(sim.state.n_postponable, GREEDY_PRIZE)
    return solve_prize_collecting(sim.state, prizes)[0]


def lazy_policy(sim: DynamicVSP) -> list[list[int]]:
    """Dispatch only must-dispatch requests."""
    prizes = np.full(sim.state.n_postponable, LAZY_PRIZE)
    return solve_prize_collecting(sim.state, prizes)[0]


def prize_policy(weights: np.ndarray) -> Policy:
    """Deterministic MILP policy: prizes theta = features @ w, then argmin Q."""
    weights = np.asarray(weights, dtype=float)

    def policy(sim: DynamicVSP) -> list[list[int]]:
        prizes = sim.actor_features() @ weights
        return solve_prize_collecting(sim.state, prizes)[0]

    return policy


def run_episode(
    sim: DynamicVSP, policy: Policy, seed: int | None, serve_final_epoch: bool
) -> float:
    """Run one episode and return its reward (negative total route duration).

    ``serve_final_epoch=False`` follows paper 02's RL evaluation loop, which
    stops once the last epoch is reached and never dispatches its requests.
    ``serve_final_epoch=True`` follows ``run_policy!`` in
    DynamicVehicleRouting.jl, used for paper 02's greedy baseline, which also
    dispatches everything in the last epoch.
    """
    sim.reset(seed)
    sim.next_epoch()
    total_cost = 0.0
    while True:
        if not serve_final_epoch and sim.is_terminated():
            break
        routes = policy(sim)
        if not sim.state.is_feasible(routes):
            raise RuntimeError(f"infeasible routes in epoch {sim.current_epoch}")
        total_cost += sim.apply_routes(routes)
        if serve_final_epoch and sim.is_terminated():
            break
        sim.next_epoch()
    return -total_cost


def evaluate(
    instances: Sequence[VSPInstance],
    policy: Policy,
    seed: int,
    serve_final_epoch: bool,
    max_requests_per_epoch: int,
) -> np.ndarray:
    return np.asarray(
        [
            run_episode(
                DynamicVSP(inst, max_requests_per_epoch, seed),
                policy,
                seed,
                serve_final_epoch,
            )
            for inst in instances
        ]
    )


def evaluate_expert(
    instances: Sequence[VSPInstance],
    seed: int,
    max_requests_per_epoch: int,
    time_limit: float | None = None,
) -> np.ndarray:
    """Anticipative (hindsight) reward per instance. Serves all epochs."""
    rewards = []
    for inst in instances:
        sim = DynamicVSP(inst, max_requests_per_epoch, seed)
        sim.draw_all_epochs(seed)
        cost, _ = anticipative_solve(sim, time_limit=time_limit)
        rewards.append(-cost)
    return np.asarray(rewards)


def delta_greedy_pct(rewards: np.ndarray, greedy_rewards: np.ndarray) -> float:
    """Improvement over greedy in percent, as in paper 02's plots (higher is better)."""
    rewards = np.asarray(rewards, dtype=float)
    greedy_rewards = np.asarray(greedy_rewards, dtype=float)
    return float(np.mean((rewards - greedy_rewards) / np.abs(greedy_rewards)) * 100)


def resolve_split_files(
    instance_dir: Path,
    split_seed: int,
    split_fractions: tuple[float, float],
    counts: dict[str, int],
    explicit: dict[str, list[str] | None],
) -> dict[str, list[Path]]:
    """Pick the instance files of each split.

    The first ``counts[split]`` explicit names are used when given; otherwise
    the first ``counts[split]`` files of a seeded shuffle-split (paper 02 uses
    10 per split).
    """
    files = list_instance_files(instance_dir)
    if not files:
        raise FileNotFoundError(f"No .txt instances found in {instance_dir}")
    by_name = {f.stem: f for f in files}
    train, val, test = split_instances(files, split_seed, split_fractions)
    default = {"train": train, "val": val, "test": test}

    out = {}
    for split, count in counts.items():
        names = explicit.get(split)
        if names:
            names = names[:count]
            missing = [n for n in names if Path(n).stem not in by_name]
            if missing:
                raise FileNotFoundError(f"{split} instances not found: {missing}")
            out[split] = [by_name[Path(n).stem] for n in names]
        else:
            out[split] = default[split][:count]
    return out


def load_splits(
    split_files: dict[str, list[Path]],
) -> dict[str, list[VSPInstance]]:
    return {
        split: [read_vsp_instance(p) for p in paths]
        for split, paths in split_files.items()
    }


class DVSPValidation:
    """Periodic deterministic evaluation during training, as in paper 02.

    The deterministic CORL policy is the MILP optimum for the current weights
    (no sampling from the candidate pool). Greedy baselines are computed once
    per split so every evaluation also reports the improvement over greedy.
    """

    def __init__(
        self,
        splits: dict[str, Sequence[VSPInstance]],
        seed: int,
        serve_final_epoch: bool,
        max_requests_per_epoch: int,
        eval_splits: Sequence[str] = ("train", "val"),
    ):
        self.splits = {name: splits[name] for name in eval_splits}
        self.kwargs = dict(
            seed=seed,
            serve_final_epoch=serve_final_epoch,
            max_requests_per_epoch=max_requests_per_epoch,
        )
        self.greedy = {
            name: evaluate(instances, greedy_policy, **self.kwargs)
            for name, instances in self.splits.items()
        }
        self.best_val = -np.inf

    def evaluate(self, weights: np.ndarray) -> dict[str, float]:
        metrics = {}
        policy = prize_policy(weights)
        for name, instances in self.splits.items():
            rewards = evaluate(instances, policy, **self.kwargs)
            metrics[f"dvsp/{name}_reward"] = float(rewards.mean())
            metrics[f"dvsp/{name}_greedy_reward"] = float(self.greedy[name].mean())
            metrics[f"dvsp/{name}_delta_greedy_pct"] = delta_greedy_pct(
                rewards, self.greedy[name]
            )
        return metrics

    def is_new_best(self, metrics: dict[str, float]) -> bool:
        val = metrics.get("dvsp/val_reward", -np.inf)
        if val > self.best_val:
            self.best_val = val
            return True
        return False
