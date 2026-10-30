"""Compare DAP results with the SRL paper (Hoppe et al., NeurIPS 2025, Table 11).

Runs the greedy and expert policies and the given trained params files on the
paper's train and test episodes (seeds 1-100 and 2001-2100 of 80 steps, as in
scripts/DAP/00_setup.jl of the SRL codebase) and prints them next to the
paper's results.

    uv run python compare_srl_paper.py params/<run id>.yaml [more params ...]

Policies trained with our SRL port (src/srl) are added with --srl:

    uv run python compare_srl_paper.py --srl runs/srl/seed_*/best.pt --average

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
params files (or SRL checkpoints) of several seeds with --average to do the same.

`train min` is the wall-clock training time of a run: `train_time_s` next to a
params file (written by scripts/run_srl_comparison.sh) or in an SRL checkpoint,
and the mean over the runs of an average. The paper's is from its Table 1.

All policies run on the same episodes, so two of them can be compared episode
by episode: --paired A B prints the mean revenue difference A - B with a
bootstrap confidence interval. An average label resamples its model seeds as
well as the episodes. With --average, the MILP and SRL averages are paired by
default.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluate import config_for_params, evaluate_dap_baselines, evaluate_params
from src.config.config_loader import load_config
from src.srl.checkpoint import read_checkpoint
from src.srl.evaluate import evaluate_srl

# Table 11 of the paper: mean episode revenue on the (train, test) episodes.
# SIL, PPO and SRL are averaged over ten random model initializations.
PAPER_DAP = {
    "expert": (569.9, 583.2),
    "greedy": (439.6, 484.1),
    "SIL": (490.9, 519.2),
    "PPO": (308.7, 313.1),
    "SRL": (529.8, 555.3),
}
# Table 1 of the paper: training time in minutes (MacBook Air M3, Julia).
PAPER_DAP_TRAIN_MIN = {"SIL": 3.0, "PPO": 5.0, "SRL": 31.0}

BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95

# First episode seed of each split in the SRL codebase.
SPLIT_FIRST_SEED = {"train": 1, "test": 2001}
SPLITS = tuple(SPLIT_FIRST_SEED)

AVERAGE_LABEL = "average"
SRL_AVERAGE_LABEL = "srl_average"


def srl_label(checkpoint: Path) -> str:
    """Label of an SRL checkpoint: its run directory, plus the file name unless
    it is the best checkpoint."""
    run = checkpoint.resolve().parent.name
    return f"srl_{run}" if checkpoint.stem == "best" else f"srl_{run}_{checkpoint.stem}"


def params_train_minutes(params_path: Path) -> float | None:
    """Training time of a params file, if its run directory recorded one."""
    time_file = params_path.resolve().parent / "train_time_s"
    return float(time_file.read_text()) / 60 if time_file.exists() else None


def srl_train_minutes(checkpoint: Path) -> float | None:
    seconds = read_checkpoint(checkpoint).get("train_time_s")
    return None if seconds is None else seconds / 60


def mean_minutes(minutes: list[float | None]) -> float | None:
    return None if None in minutes else float(np.mean(minutes))


def paired_difference(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean per-episode difference of policies a - b and its bootstrap confidence
    interval. a and b hold one row of per-episode revenue per model seed; the
    bootstrap resamples the episodes (shared by a and b) and each one's seeds."""
    n_episodes = a.shape[1]
    diffs = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        episodes = rng.integers(n_episodes, size=n_episodes)
        a_runs = rng.integers(len(a), size=len(a))
        b_runs = rng.integers(len(b), size=len(b))
        diffs[i] = (
            a[a_runs][:, episodes].mean() - b[b_runs][:, episodes].mean()
        )
    tail = 100 * (1 - CONFIDENCE) / 2
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(a.mean() - b.mean()), float(low), float(high)


def print_paired(
    pairs: list[tuple[str, str]],
    revenue: dict[str, np.ndarray],
    members: dict[str, list[str]],
) -> None:
    rng = np.random.default_rng(0)
    for a_label, b_label in pairs:
        a = np.array([revenue[label] for label in members.get(a_label, [a_label])])
        b = np.array([revenue[label] for label in members.get(b_label, [b_label])])
        mean, low, high = paired_difference(a, b, rng)
        wins = int((a.mean(0) > b.mean(0)).sum())
        print(
            f"\n{a_label} - {b_label}: {mean:+.1f} revenue per episode "
            f"({100 * CONFIDENCE:.0f} % CI {low:+.1f} to {high:+.1f}); "
            f"{a_label} is better on {wins} of {a.shape[1]} episodes"
        )


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
    train_min: float | None = None,
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
        "train_min": train_min,
    }


def summarize_split(
    split: str,
    revenue: dict[str, np.ndarray],
    train_minutes: dict[str, float | None],
) -> list[dict]:
    split_idx = SPLITS.index(split)
    paper_greedy = PAPER_DAP["greedy"][split_idx]
    paper_expert = PAPER_DAP["expert"][split_idx]
    rows = [
        summary_row(
            "paper",
            label,
            means[split_idx],
            paper_greedy,
            paper_expert,
            train_min=PAPER_DAP_TRAIN_MIN.get(label),
        )
        for label, means in PAPER_DAP.items()
    ]

    greedy = revenue["greedy"]
    expert_mean = revenue["expert"].mean()
    for label, episodes in revenue.items():
        gain = (episodes - greedy) / np.where(greedy == 0, np.nan, greedy)
        rows.append(
            summary_row(
                "ours",
                label,
                episodes.mean(),
                greedy.mean(),
                expert_mean,
                gain,
                train_minutes.get(label),
            )
        )
    return rows


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}"


def format_minutes(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


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
        f"  {'train min':>9}"
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
            f"  {format_minutes(row['train_min']):>9}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", nargs="*", type=Path, help="Saved DAP params files.")
    parser.add_argument(
        "--srl",
        nargs="+",
        type=Path,
        default=[],
        help="Checkpoints of our SRL port (src/srl), e.g. runs/srl/seed_*/best.pt.",
    )
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
        help="Also report the per-episode mean over all params files, and over "
        "all SRL checkpoints, as the paper does over its ten model seeds.",
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
    parser.add_argument(
        "--paired",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        help="Print the per-episode revenue difference A - B of two policies "
        "(labels as in the table). Can be repeated.",
    )
    parser.add_argument("--csv", type=Path, help="Write per-episode results here.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_config = load_config(project_root / "config.yaml")
    runs = []
    train_minutes = {}
    for params_path in args.params:
        config = config_for_params(params_path.resolve(), base_config)
        if config.problem != "dap":
            parser.error(f"{params_path} was trained on '{config.problem}', not 'dap'")
        runs.append((params_path.stem, config))
        train_minutes[params_path.stem] = params_train_minutes(params_path)
    srl_runs = [(srl_label(path), path.resolve()) for path in args.srl]
    for label, checkpoint in srl_runs:
        train_minutes[label] = srl_train_minutes(checkpoint)
    labels = [label for label, _ in runs + srl_runs]
    if len(set(labels)) != len(labels):
        parser.error(f"two runs share a label: {labels}")
    if args.average and len(runs) < 2 and len(srl_runs) < 2:
        parser.error("--average needs at least two params files or SRL checkpoints")

    # Members of each average, for its training time and the paired bootstrap.
    members = {}
    for average_label, averaged in ((AVERAGE_LABEL, runs), (SRL_AVERAGE_LABEL, srl_runs)):
        if args.average and len(averaged) >= 2:
            members[average_label] = [label for label, _ in averaged]
            train_minutes[average_label] = mean_minutes(
                [train_minutes[label] for label, _ in averaged]
            )
    pairs = args.paired or []
    if not pairs and AVERAGE_LABEL in members and SRL_AVERAGE_LABEL in members:
        pairs = [(AVERAGE_LABEL, SRL_AVERAGE_LABEL)]
    known_labels = set(labels) | set(members) | {"greedy", "expert"}
    for pair in pairs:
        if unknown := set(pair) - known_labels:
            parser.error(f"--paired: unknown labels {sorted(unknown)}")

    split_seeds = {
        split: list(range(first_seed, first_seed + args.episodes))
        for split, first_seed in SPLIT_FIRST_SEED.items()
    }
    n_jobs = (len(runs) + len(srl_runs)) * len(args.splits)
    pool = ProcessPoolExecutor(max_workers=max(1, min(args.jobs, n_jobs)))
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
    } | {
        (split, label): pool.submit(
            evaluate_srl,
            label,
            checkpoint,
            split_seeds[split],
            base_config.dap_env.max_steps,
        )
        for split in args.splits
        for label, checkpoint in srl_runs
    }

    all_rows = []
    for split in args.splits:
        print(f"\n== {split} ==")
        rows = evaluate_dap_baselines(
            base_config, split_seeds[split], base_config.dap_env.max_steps
        )
        for label, _ in runs + srl_runs:
            rows.extend(evaluations[split, label].result())
        revenue = revenue_by_run(rows, args.metric)
        for average_label, labels in members.items():
            revenue[average_label] = np.mean([revenue[label] for label in labels], axis=0)
        print_split(
            split, summarize_split(split, revenue, train_minutes), args.episodes
        )
        print_paired(pairs, revenue, members)
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
