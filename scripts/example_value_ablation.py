"""Test the value-term ablation (example or portfolio problem) trained by
scripts/run_example_value_ablation.sh.

    uv run python scripts/example_value_ablation.py runs/example_value_<timestamp>

For every model seed (one problem instance each) it evaluates, on the same test
episodes, the untrained and the best trained parameters with and without the
value term v, and the policy that always takes the zero action. Whether higher or lower reward is better is read from
validation.maximize_reward of the runs' config. The summary gives
the mean over seeds with the standard error of the seed means, and the paired
differences between the variants with a paired t-test.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import config_for_params, evaluate, evaluate_params  # noqa: E402
from train import build_model_and_env  # noqa: E402
from src.config.config_loader import load_config  # noqa: E402

VARIANTS = ("v", "no_v")


def runs_of_seed(seed_dir: Path) -> list[tuple[str, object]]:
    """(label, config) of the untrained and trained runs of one seed."""
    runs = [("zero_action", load_config(seed_dir / "v" / "config.yaml"))]
    for variant in VARIANTS:
        variant_dir = seed_dir / variant
        config = load_config(variant_dir / "config.yaml")
        runs.append((f"untrained_{variant}", config.model_copy(update={"load": False})))
        best = sorted(variant_dir.glob("*_best.yaml"))
        if best:
            runs.append((f"trained_{variant}", config_for_params(best[0], config)))
        else:
            print(f"{variant_dir} has no best params, leaving it out")
    return runs


def evaluate_run(seed: int, label: str, config, test_seeds: list[int]) -> list[dict]:
    if label == "zero_action":
        _, env = build_model_and_env(config, PROJECT_ROOT)
        zero = np.zeros(env.action_space.shape, dtype=np.int32)
        rows = evaluate(label, env, lambda: zero, test_seeds, 10)
    else:
        rows = evaluate_params(label, config, PROJECT_ROOT, test_seeds, 10)
    return [{"model_seed": seed, **row} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--first-test-seed", type=int, default=10_000)
    args = parser.parse_args()

    test_seeds = list(range(args.first_test_seed, args.first_test_seed + args.episodes))
    jobs = []
    for seed_dir in sorted(args.run_dir.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        jobs.extend((seed, label, config) for label, config in runs_of_seed(seed_dir))

    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(evaluate_run, *job, test_seeds) for job in jobs]
        rows = [row for future in futures for row in future.result()]

    with (args.run_dir / "test_episodes.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Mean test cost of every (seed, run).
    seed_means: dict[str, dict[int, float]] = {}
    for row in rows:
        seed_means.setdefault(row["run"], {}).setdefault(row["model_seed"], [])
        seed_means[row["run"]][row["model_seed"]].append(row["ep_reward"])
    seed_means = {
        run: {seed: float(np.mean(r)) for seed, r in by_seed.items()}
        for run, by_seed in seed_means.items()
    }
    labels = [
        f"{kind}_{variant}" for kind in ("trained", "untrained") for variant in VARIANTS
    ]
    labels = [l for l in labels + ["zero_action"] if l in seed_means]
    maximize = load_config(next(args.run_dir.glob("seed_*/v/config.yaml"))).validation.maximize_reward
    sign = 1 if maximize else -1
    seeds = sorted(set.intersection(*(set(seed_means[label]) for label in labels)))

    lines = [f"Test reward per episode ({'higher' if maximize else 'lower'} is better), {len(test_seeds)} episodes, "
             f"{len(seeds)} model seeds"]
    lines.append("")
    lines.append("seed  " + "".join(f"{label:>14}" for label in labels))
    for seed in seeds:
        lines.append(
            f"{seed:<6}" + "".join(f"{seed_means[l][seed]:>14.1f}" for l in labels)
        )
    lines.append("")
    lines.append(f"{'run':<16}{'mean':>10}{'SE':>10}")
    for label in labels:
        values = np.array([seed_means[label][s] for s in seeds])
        lines.append(f"{label:<16}{values.mean():>10.1f}{stats.sem(values):>10.1f}")
    lines.append("")
    lines.append(f"Paired differences over seeds (a - b; {'positive' if maximize else 'negative'} means a is better)")
    pairs = [
        ("trained_v", "trained_no_v"),
        ("trained_v", "untrained_v"),
        ("trained_no_v", "untrained_no_v"),
        ("untrained_v", "untrained_no_v"),
        ("trained_v", "zero_action"),
        ("trained_no_v", "zero_action"),
    ]
    for a, b in pairs:
        if a not in seed_means or b not in seed_means:
            continue
        diff = np.array([seed_means[a][s] - seed_means[b][s] for s in seeds])
        p = stats.ttest_rel([seed_means[a][s] for s in seeds],
                            [seed_means[b][s] for s in seeds]).pvalue
        wins = int((sign * diff > 0).sum())
        lines.append(
            f"{a} - {b}: {diff.mean():.1f} +/- {stats.sem(diff):.1f} (SE), "
            f"p = {p:.3f}, a better on {wins}/{len(seeds)} seeds"
        )
    summary = "\n".join(lines)
    print(summary)
    (args.run_dir / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
