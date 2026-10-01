"""DAP baselines and CORL evaluation in the format of paper 02's Table 11.

Evaluates greedy, the myopic expert and, optionally, trained CORL policies on
paper 02's episodes: train = seeds 1-100, test = seeds 2001-2100. Rewards are
the total revenue per episode (higher is better). With several --params
files (e.g. one per training seed), CORL is reported as mean +- std over
them.

Usage:
  uv run python scripts/dap_baselines.py
  uv run python scripts/dap_baselines.py --params params/dap_dap_corl_bnb_seed_*_best.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dap.evaluation import (  # noqa: E402
    TEST_SEEDS,
    TRAIN_SEEDS,
    evaluate,
    expert_policy,
    greedy_policy,
    milp_policy,
)
from src.dap.milp import n_params  # noqa: E402

# Paper 02, Table 11 (DAP, train / test).
PAPER02 = {
    "Expert": (569.9, 583.2),
    "Greedy": (439.6, 484.1),
    "SIL": (490.9, 519.2),
    "PPO": (308.7, 313.1),
    "SRL": (529.8, 555.3),
}


def n_pieces_of(theta: np.ndarray) -> int:
    for j in range(0, 64):
        if n_params(j) == theta.size:
            return j
    raise ValueError(f"cannot infer the number of value pieces from {theta.size} parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", nargs="*", default=[], help="YAML files with key 'theta'.")
    args = parser.parse_args()

    splits = {"train": TRAIN_SEEDS, "test": TEST_SEEDS}
    greedy = {s: evaluate(greedy_policy, seeds) for s, seeds in splits.items()}
    expert = {s: evaluate(expert_policy, seeds) for s, seeds in splits.items()}
    corl = {s: [] for s in splits}
    for path in args.params:
        theta = np.array(yaml.safe_load(open(path))["theta"], dtype=float)
        policy = milp_policy(theta, n_pieces_of(theta))
        for s, seeds in splits.items():
            corl[s].append(evaluate(policy, seeds).mean())
        print(f"  {Path(path).name}: train {corl['train'][-1]:.1f}  test {corl['test'][-1]:.1f}")

    def pct(value, split):
        return (value - greedy[split].mean()) / greedy[split].mean() * 100

    def gap(value, split):
        g, e = greedy[split].mean(), expert[split].mean()
        return (value - g) / (e - g) * 100

    print("\nMean revenue per episode (100 episodes), improvement over greedy and share")
    print("of the greedy->expert gap closed:")
    for split in splits:
        g, e = greedy[split].mean(), expert[split].mean()
        line = f"  {split:5s} greedy {g:6.1f} | expert {e:6.1f} ({pct(e, split):+5.1f}%)"
        if corl[split]:
            values = np.array(corl[split])
            spread = f" ± {values.std(ddof=1):.1f}" if len(values) > 1 else ""
            line += (f" | CORL {values.mean():6.1f}{spread} ({pct(values.mean(), split):+5.1f}%, "
                     f"closes {gap(values.mean(), split):.0f}% of gap, n={len(values)})")
        print(line)

    print("\nPaper 02, Table 11 (train / test), with the same relative measures:")
    for name, (train, test) in PAPER02.items():
        pg_tr, pg_te = PAPER02["Greedy"]
        pe_tr, pe_te = PAPER02["Expert"]
        print(f"  {name:6s} {train:6.1f} / {test:6.1f}   vs greedy {(train - pg_tr) / pg_tr * 100:+5.1f}% / "
              f"{(test - pg_te) / pg_te * 100:+5.1f}%   gap closed {(train - pg_tr) / (pe_tr - pg_tr) * 100:4.0f}% / "
              f"{(test - pg_te) / (pe_te - pg_te) * 100:4.0f}%")
    print("Episodes differ from paper 02 (numpy vs Julia random numbers), so compare the "
          "relative measures.")


if __name__ == "__main__":
    main()
