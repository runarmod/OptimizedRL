"""How often the DAP MILP has a single candidate action, with and without the
pair terms (McCormick constraints).

For every formulation and parameter vector, episodes are run with the most
probable action (as in validation), and at every state it records:
  - whether the root LP relaxation is integral, and how many x_i are fractional,
  - the number of candidates in the scip_brute pool (the actions PPO samples from),
  - the solve times of the pool and of the best action.

    uv run python scripts/dap_formulation_stats.py --params params/nn2jdgn2.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_config  # noqa: E402
from src.gym_envs import dap_env  # noqa: E402
from src.models import dap_model  # noqa: E402
from src.utils.checkpoint import load_model_params  # noqa: E402
from train import build_dap_env, build_solver, sanitize_env_action  # noqa: E402

INTEGRAL_TOL = 1e-6


def root_lp(node: dict) -> np.ndarray:
    res = linprog(
        node["c"],
        A_ub=node["A_ub"],
        b_ub=node["b_ub"],
        A_eq=node["A_eq"],
        b_eq=node["b_eq"],
        bounds=node["bounds"],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def make_model(theta, config, pairwise):
    cfg = config.dap_env
    return dap_model.DapModel(
        theta.copy(),
        n_items=cfg.n_items,
        assortment_size=cfg.assortment_size,
        n_value_pieces=config.model.n_value_func,
        pairwise=pairwise,
    )


def run(model, solver, env, seeds, max_steps):
    rows = []
    for seed in seeds:
        np.random.seed(seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            model.update_from_environment(env)
            node = model.get_LP_formulation()

            x = root_lp(node)[: model.n_items]
            n_frac = int(np.sum(np.abs(x - np.round(x)) > INTEGRAL_TOL))

            t0 = time.perf_counter()
            pool = solver.solve(node) or []
            t_pool = time.perf_counter() - t0

            t0 = time.perf_counter()
            action = solver.solve_best(node, model)
            t_best = time.perf_counter() - t0

            rows.append((n_frac, len(pool), t_pool, t_best))
            _, _, terminated, _, _ = env.step(sanitize_env_action(env, action))
            if terminated:
                break
    return np.array(rows, dtype=float)


def summarize(name, rows):
    n_frac, n_pool, t_pool, t_best = rows.T
    hist = np.bincount(n_pool.astype(int))
    top = ", ".join(f"{k}:{v}" for k, v in enumerate(hist) if v)
    print(
        f"{name:<34} states={len(rows):>4}  root LP integral={np.mean(n_frac == 0):6.1%}"
        f"  mean frac x={n_frac.mean():4.2f}  pool size 1={np.mean(n_pool == 1):6.1%}"
        f"  mean pool={n_pool.mean():5.2f}  max pool={int(n_pool.max())}"
        f"  t_pool={1000 * t_pool.mean():6.1f}ms  t_best={1000 * t_best.mean():6.1f}ms"
    )
    print(f"{'':<34} pool size histogram {{size:count}} = {top}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dap_srl.yaml")
    parser.add_argument("--params", nargs="*", default=[])
    parser.add_argument("--init-seeds", nargs="*", type=int, default=[5])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 1001, 1002])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--pieces", type=int, default=None, help="Override model.n_value_func."
    )
    parser.add_argument(
        "--formulations",
        nargs="+",
        choices=["mccormick", "no-pairs"],
        default=["mccormick", "no-pairs"],
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.pieces is not None:
        config.model.n_value_func = args.pieces
    solver = build_solver(config)
    env = build_dap_env(config)
    size = 2 * dap_model.N_FEATURES + config.model.n_value_func * (
        2 * dap_model.N_FEATURES + 1
    )

    sources = []
    for seed in args.init_seeds:
        np.random.seed(seed)
        sources.append((f"init seed {seed}", np.random.uniform(0, 0.1, size=size)))
    for path in args.params:
        model = make_model(np.zeros(size), config, True)
        load_model_params(Path(path), model)
        sources.append((Path(path).stem, model.theta))

    for label, theta in sources:
        for pairwise in [f == "mccormick" for f in args.formulations]:
            name = (
                f"{label} / L={config.model.n_value_func} / "
                f"{'McCormick' if pairwise else 'no pairs'}"
            )
            rows = run(make_model(theta, config, pairwise), solver, env, args.seeds, args.max_steps)
            summarize(name, rows)


if __name__ == "__main__":
    main()
