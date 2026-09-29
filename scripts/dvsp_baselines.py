"""DVSP baselines and policy evaluation in the format of paper 02's Table 11.

Evaluates greedy, lazy and the anticipative expert on the train and test
instances, and optionally a trained CORL policy (weights saved by train.py).
Rewards are minus the total route duration in hours, averaged over instances,
with one fixed request stream per instance (seed ``dvsp.scenario_seed``).

Both episode protocols are reported, because paper 02 mixes them: its RL
evaluation stops before the last epoch ("skip-final"), while its greedy and
expert baselines also dispatch the last epoch ("serve-final").

Usage:
  uv run python scripts/dvsp_baselines.py [--config config_dvsp.yaml]
      [--params params/dvsp_<name>_best.yaml] [--splits train test]
"""

import argparse
import time

import numpy as np
import yaml
from _dvsp_common import load_dvsp_config, load_dvsp_splits

from src.dvsp.evaluation import (
    delta_greedy_pct,
    evaluate,
    evaluate_expert,
    greedy_policy,
    lazy_policy,
    prize_policy,
)

PAPER02_TABLE11 = {
    "Expert": (-30.1, -25.5),
    "Greedy": (-35.1, -30.0),
    "SIL": (-31.8, -27.2),
    "PPO": (-37.4, -32.5),
    "SRL": (-31.9, -27.3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config_dvsp.yaml")
    parser.add_argument("--params", default=None, help="YAML file with key 'w'.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--no-expert", action="store_true")
    parser.add_argument("--expert-time-limit", type=float, default=None)
    args = parser.parse_args()

    config = load_dvsp_config(args.config)
    cfg = config.dvsp
    splits = load_dvsp_splits(config, tuple(args.splits))
    common = dict(seed=cfg.scenario_seed, max_requests_per_epoch=cfg.max_requests_per_epoch)

    policies = {"Greedy": greedy_policy, "Lazy": lazy_policy}
    if args.params:
        with open(args.params) as f:
            weights = np.array(yaml.safe_load(f)["w"], dtype=float)
        policies["CORL"] = prize_policy(weights)

    results = {}  # (policy, protocol, split) -> rewards
    for split, instances in splits.items():
        for serve_final in (False, True):
            protocol = "serve-final" if serve_final else "skip-final"
            for name, policy in policies.items():
                start = time.time()
                rewards = evaluate(instances, policy, serve_final_epoch=serve_final, **common)
                results[(name, protocol, split)] = rewards
                print(f"  {name:7s} {protocol:11s} {split:5s} {rewards.mean():8.2f}  ({time.time() - start:.1f}s)")
        if not args.no_expert:
            start = time.time()
            rewards = evaluate_expert(instances, time_limit=args.expert_time_limit, **common)
            results[("Expert", "serve-final", split)] = rewards
            print(f"  Expert  serve-final {split:5s} {rewards.mean():8.2f}  ({time.time() - start:.1f}s)")

    print()
    print("Mean episode reward (higher is better) and improvement over greedy")
    print("under the same protocol, in %:")
    header = f"{'policy':8s} {'protocol':12s}" + "".join(f"{s:>20s}" for s in splits)
    print(header)
    print("-" * len(header))
    for (name, protocol) in dict.fromkeys((k[0], k[1]) for k in results):
        row = f"{name:8s} {protocol:12s}"
        for split in splits:
            rewards = results.get((name, protocol, split))
            if rewards is None:
                row += f"{'':>20s}"
                continue
            delta = delta_greedy_pct(rewards, results[("Greedy", protocol, split)])
            row += f"{rewards.mean():>11.2f} ({delta:+5.1f}%)"
        print(row)

    print()
    print("Paper 02, Table 11 (DVSP, train / test):")
    for name, (train, test) in PAPER02_TABLE11.items():
        print(f"  {name:7s} {train:7.1f} / {test:7.1f}")
    print(
        "Instances and request streams differ from paper 02 (Julia RNG), so compare "
        "the relative numbers (improvement over greedy), not absolute rewards."
    )


if __name__ == "__main__":
    main()
