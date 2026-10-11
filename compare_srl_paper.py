"""Compare DAP results with the SRL paper (Hoppe et al., NeurIPS 2025, Table 11).

Runs the greedy and expert policies and the given trained params files on the
paper's train and test episodes (seeds 1-100 and 2001-2100 of 80 steps, as in
scripts/DAP/00_setup.jl of the SRL codebase) and prints them next to the
paper's results.

    uv run python compare_srl_paper.py params/<run id>.yaml [more params ...]

Our simulator draws its instances with numpy instead of Julia's MersenneTwister,
so the same seed gives a different instance and absolute revenues do not match
the paper. Compare the columns relative to greedy and expert instead:

    gain_greedy    (mean - mean greedy) / mean greedy, from the means, as the
                   paper's Table 11 allows.
    gain_greedy_i  mean of the per-episode (revenue - greedy) / greedy, the dot
                   in the paper's Figure 5 box plots (not tabulated in the paper).
    expert_gap     (mean - mean greedy) / (mean expert - mean greedy): the share
                   of the expert's gain over greedy that a policy reaches.

The paper averages ten trained models per episode before comparing. Pass the
params files of several seeds with --average to do the same.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluate import config_for_params, evaluate_dap_baselines, evaluate_params
from src.config.config_loader import load_config

# Table 11 of the paper: mean episode revenue on the (train, test) episodes.
# SIL, PPO and SRL are averaged over ten random model initializations.
PAPER_DAP = {
    "expert": (569.9, 583.2),
    "greedy": (439.6, 484.1),
    "SIL": (490.9, 519.2),
    "PPO": (308.7, 313.1),
    "SRL": (529.8, 555.3),
}

# First episode seed of each split in the SRL codebase (validation is 1001).
SPLIT_FIRST_SEED = {"train": 1, "test": 2001}
SPLITS = tuple(SPLIT_FIRST_SEED)

AVERAGE_LABEL = "average"


def revenue_by_run(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    """Per-episode revenue of every run, in seed order."""
    runs: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        runs.setdefault(row["run"], []).append((row["seed"], row[metric]))
    return {
        label: np.array([value for _, value in sorted(episodes)])
        for label, episodes in runs.items()
    }


def summary_row(
    source: str,
    label: str,
    mean: float,
    greedy_mean: float,
    expert_mean: float,
    gain_per_episode: np.ndarray | None = None,
) -> dict:
    return {
        "source": source,
        "policy": label,
        "mean": mean,
        "gain_greedy": 100 * (mean - greedy_mean) / greedy_mean,
        "gain_greedy_i": (
            None if gain_per_episode is None else 100 * gain_per_episode.mean()
        ),
        "expert_gap": 100 * (mean - greedy_mean) / (expert_mean - greedy_mean),
    }


def summarize_split(split: str, revenue: dict[str, np.ndarray]) -> list[dict]:
    split_idx = SPLITS.index(split)
    paper_greedy = PAPER_DAP["greedy"][split_idx]
    paper_expert = PAPER_DAP["expert"][split_idx]
    rows = [
        summary_row("paper", label, means[split_idx], paper_greedy, paper_expert)
        for label, means in PAPER_DAP.items()
    ]

    greedy = revenue["greedy"]
    expert_mean = revenue["expert"].mean()
    for label, episodes in revenue.items():
        gain = (episodes - greedy) / np.where(greedy == 0, np.nan, greedy)
        rows.append(
            summary_row(
                "ours", label, episodes.mean(), greedy.mean(), expert_mean, gain
            )
        )
    return rows


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}"


def print_split(split: str, rows: list[dict], n_episodes: int) -> None:
    first_seed = SPLIT_FIRST_SEED[split]
    print(
        f"\nDAP {split}: seeds {first_seed}-{first_seed + n_episodes - 1}"
        " (paper: 100 episodes)"
    )
    label_width = max(len(row["policy"]) for row in rows)
    header = (
        f"{'source':<6}  {'policy':<{label_width}}  {'mean rev':>9}"
        f"  {'gain_greedy %':>13}  {'gain_greedy_i %':>15}  {'expert_gap %':>12}"
    )
    print(header)
    print("-" * len(header))
    previous_source = rows[0]["source"]
    for row in rows:
        if row["source"] != previous_source:
            print()
            previous_source = row["source"]
        print(
            f"{row['source']:<6}  {row['policy']:<{label_width}}  {row['mean']:>9.1f}"
            f"  {format_percent(row['gain_greedy']):>13}"
            f"  {format_percent(row['gain_greedy_i']):>15}"
            f"  {format_percent(row['expert_gap']):>12}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", nargs="*", type=Path, help="Saved DAP params files.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per split, from its first seed (paper: 100).",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--average",
        action="store_true",
        help="Also report the per-episode mean over all params files, as the "
        "paper does over its ten model seeds.",
    )
    parser.add_argument(
        "--metric",
        choices=("ep_reward", "expected_revenue"),
        default="ep_reward",
        help="Realized revenue (as in the paper) or its expectation under the "
        "customer model, which has less sampling noise.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Params files and splits evaluated in parallel.",
    )
    parser.add_argument("--csv", type=Path, help="Write per-episode results here.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_config = load_config(project_root / "config.yaml")
    runs = []
    for params_path in args.params:
        config = config_for_params(params_path.resolve(), base_config)
        if config.problem != "dap":
            parser.error(f"{params_path} was trained on '{config.problem}', not 'dap'")
        runs.append((params_path.stem, config))
    if args.average and len(runs) < 2:
        parser.error("--average needs at least two params files")

    split_seeds = {
        split: list(range(first_seed, first_seed + args.episodes))
        for split, first_seed in SPLIT_FIRST_SEED.items()
    }
    pool = ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(runs) * 2)))
    evaluations = {
        (split, label): pool.submit(
            evaluate_params,
            label,
            config,
            project_root,
            split_seeds[split],
            config.dap_env.max_steps,
        )
        for split in args.splits
        for label, config in runs
    }

    all_rows = []
    for split in args.splits:
        print(f"\n== {split} ==")
        rows = evaluate_dap_baselines(
            base_config, split_seeds[split], base_config.dap_env.max_steps
        )
        for label, _ in runs:
            rows.extend(evaluations[split, label].result())
        revenue = revenue_by_run(rows, args.metric)
        if args.average:
            revenue[AVERAGE_LABEL] = np.mean(
                [revenue[label] for label, _ in runs], axis=0
            )
        print_split(split, summarize_split(split, revenue), args.episodes)
        all_rows.extend({"split": split, **row} for row in rows)
    pool.shutdown()

    if args.metric != "ep_reward":
        print(f"\n'ours' rows use {args.metric}; the paper reports realized revenue.")

    if args.csv:
        fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
        with args.csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-episode results written to {args.csv}")


if __name__ == "__main__":
    main()
