"""Check whether the DVSP MILP collapses to its root in branch-and-bound.

The prize-collecting VSP has a network-flow constraint matrix, so its LP
relaxation should be integral. Then SCIP never branches and the CORL policy,
a softmax over B&B nodes, has (almost) a single candidate. This script
measures that on real epoch states and compares the generic SCIP candidate
pool with the dispatch-branching enumeration pool.

Usage: uv run python scripts/dvsp_check_integrality.py [--config config_dvsp.yaml]
"""

import argparse
from collections import Counter

import numpy as np
from _dvsp_common import load_dvsp_config, load_dvsp_splits
from scipy.optimize import linprog

from src.dvsp.evaluation import greedy_policy
from src.dvsp.simulator import DynamicVSP
from src.models.dvsp_model import DVSPModel, initial_weights
from src.solvers.enumeration import BranchEnumerationSolver
from src.solvers.scip import SCIPSolver


class EpochSnapshot:
    """Frozen copy of one simulator epoch, readable by DVSPModel."""

    def __init__(self, sim: DynamicVSP):
        self.state = sim.state_summary()
        self.epoch_state = sim.state
        self.features = sim.actor_features()

    def actor_features(self):
        return self.features


def collect_states(instances, max_requests, seed, max_states):
    """Epoch states along greedy trajectories, so later epochs are included."""
    views = []
    for inst in instances:
        sim = DynamicVSP(inst, max_requests, seed)
        sim.reset(seed)
        sim.next_epoch()
        while not sim.is_terminated() and len(views) < max_states:
            if len(sim.state.arcs):
                views.append(EpochSnapshot(sim))
            routes = greedy_policy(sim)
            sim.apply_routes(routes)
            sim.next_epoch()
    return views


def is_integral(x, tol=1e-6):
    return bool(np.all(np.abs(x - np.round(x)) <= tol))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config_dvsp.yaml")
    parser.add_argument("--max-states", type=int, default=40)
    parser.add_argument("--weight-seeds", type=int, default=3)
    args = parser.parse_args()

    config = load_dvsp_config(args.config)
    cfg = config.dvsp
    instances = load_dvsp_splits(config, ("train",))["train"]
    views = collect_states(
        instances, cfg.max_requests_per_epoch, cfg.scenario_seed, args.max_states
    )

    scip_solver = SCIPSolver(verbose=False)
    enum_solver = BranchEnumerationSolver(max_pool_size=16, max_depth=1)
    n_lp = n_lp_integral = 0
    scip_sizes, enum_sizes = [], []
    scip_status = Counter()
    enum_integral = enum_total = 0

    for seed in range(args.weight_seeds):
        model = DVSPModel(initial_weights("glorot", seed))
        for view in views:
            model.update_from_environment(view)
            node = model.get_LP_formulation()

            for method in ("highs-ds", "highs-ipm"):
                res = linprog(
                    node["c"],
                    A_ub=node["A_ub"],
                    b_ub=node["b_ub"],
                    A_eq=node["A_eq"],
                    b_eq=node["b_eq"],
                    bounds=node["bounds"],
                    method=method,
                )
                n_lp += 1
                n_lp_integral += int(res.success and is_integral(res.x))

            pool = scip_solver.solve(node)
            scip_sizes.append(len(pool))
            scip_status.update(entry.get("status") for entry in pool)

            pool = enum_solver.solve(node)
            enum_sizes.append(len(pool))
            enum_total += len(pool)
            enum_integral += sum(is_integral(e["x"]) for e in pool)

    print(f"epoch states: {len(views)} x {args.weight_seeds} weight seeds")
    print(f"LP relaxation integral: {n_lp_integral}/{n_lp} (dual simplex and IPM)")
    print(
        f"SCIP pool size: mean {np.mean(scip_sizes):.2f}, "
        f"min {min(scip_sizes)}, max {max(scip_sizes)}"
    )
    print(f"SCIP pool composition by status: {dict(scip_status)}")
    print(
        f"Enumeration pool size: mean {np.mean(enum_sizes):.2f}, "
        f"min {min(enum_sizes)}, max {max(enum_sizes)}; "
        f"integral candidates {enum_integral}/{enum_total}"
    )


if __name__ == "__main__":
    main()
