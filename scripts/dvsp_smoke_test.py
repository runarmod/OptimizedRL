"""Fast correctness checks for the DVSP integration (no wandb, ~1 minute).

Usage: uv run python scripts/dvsp_smoke_test.py [--config config_dvsp.yaml]
"""

import argparse

import numpy as np
from _dvsp_common import PROJECT_ROOT, load_dvsp_config
from scipy.optimize import linprog

import train
from src.dvsp.evaluation import (
    evaluate,
    evaluate_expert,
    greedy_policy,
    lazy_policy,
    run_episode,
)
from src.dvsp.simulator import DynamicVSP, routes_cost, routes_from_arc_values
from src.dvsp.solvers import PrizeCollectingLP
from src.models.dvsp_model import DVSPModel, initial_weights
from src.solvers.enumeration import BranchEnumerationSolver


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok  {message}")


def lp_value(node):
    res = linprog(
        node["c"],
        A_ub=node["A_ub"],
        b_ub=node["b_ub"],
        A_eq=node["A_eq"],
        b_eq=node["b_eq"],
        bounds=node["bounds"],
        method="highs-ds",
    )
    return res.fun, res.x


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config_dvsp.yaml")
    args = parser.parse_args()

    config = load_dvsp_config(args.config)
    config = config.model_copy(
        update={
            "wandb_mode": "disabled",
            "total_iters": 3,
            "dvsp": config.dvsp.model_copy(
                update={
                    "nb_train_instances": 3,
                    "nb_val_instances": 2,
                    "eval_every": 1,
                    "save_best": False,
                }
            ),
            "ppo": config.ppo.model_copy(update={"rollout_iters": 20}),
        }
    )
    cfg = config.dvsp

    # --- environment and model -------------------------------------------
    model, env, validation = train.build_dvsp(config, PROJECT_ROOT)
    instance = env.sims[0].instance
    check(env.state.shape == (33,), "state summary has 33 entries")

    rewards = [
        run_episode(DynamicVSP(instance, 10, 0), policy, 0, final)
        for policy in (greedy_policy, lazy_policy)
        for final in (False, True)
    ]
    check(all(r < 0 for r in rewards), f"baseline episodes run: {np.round(rewards, 2)}")
    expert = evaluate_expert([instance], 0, cfg.max_requests_per_epoch)[0]
    greedy_full = evaluate([instance], greedy_policy, 0, True, 10)[0]
    check(expert >= greedy_full - 1e-6, f"expert {expert:.2f} >= greedy {greedy_full:.2f}")

    # --- LP integrality and the envelope-theorem gradient ----------------
    rng = np.random.default_rng(0)
    model = DVSPModel(initial_weights("glorot", 1))
    env.reset(seed=1)
    n_checked = 0
    while n_checked < 5:
        model.update_from_environment(env)
        if model.lp.n_arcs and len(model.lp.postponable):
            node = model.get_LP_formulation()
            fun, x = lp_value(node)
            check(np.allclose(x, np.round(x), atol=1e-7), "LP relaxation is integral")
            grad = model.lagrange_gradient(x, env.state, None, None)
            # Q is linear in w for fixed y: exact directional derivative.
            direction = rng.normal(size=14)
            w0 = model.w.copy()
            model.set_policy_params(w0 + 1e-3 * direction)
            c_shift = model.get_LP_formulation()["c"]
            model.set_policy_params(w0)
            fd = (c_shift @ x - fun) / 1e-3
            check(np.isclose(fd, grad @ direction, rtol=1e-5, atol=1e-6),
                  f"dQ/dw matches finite difference ({fd:.4f})")
            n_checked += 1
        state, _, terminated, _, _ = env.step(
            np.round(lp_value(model.get_LP_formulation())[1])
            if model.lp.n_arcs else np.zeros(0)
        )
        if terminated:
            env.reset()

    # --- candidate pool --------------------------------------------------
    solver = BranchEnumerationSolver(max_pool_size=16, max_depth=1)
    env.reset(seed=2)
    model.update_from_environment(env)
    node = model.get_LP_formulation()
    pool = solver.solve(node)
    funs = [entry["fun"] for entry in pool]
    check(len(pool) > 1, f"enumeration pool has {len(pool)} candidates")
    check(np.isclose(funs[0], min(funs)), "root candidate is the MILP optimum")
    state = env.epoch_state
    for entry in pool:
        routes = routes_from_arc_values(state.arcs, entry["x"])
        assert state.is_feasible(routes), "candidate routes infeasible"
        assert np.isclose(entry["fun"], node["c"] @ entry["x"])
    check(True, "every candidate decodes to feasible routes with exact Q")
    chosen = pool[-1]
    routes = routes_from_arc_values(state.arcs, chosen["x"])
    _, reward, _, _, info = env.step(chosen["x"])
    check(np.isclose(-reward, routes_cost(routes, state.duration)),
          f"env reward equals minus route duration ({reward:.3f})")

    # --- short end-to-end PPO training -----------------------------------
    model, env, validation = train.build_dvsp(config, PROJECT_ROOT)
    solver = train.build_solver(config)
    agent = train.build_training_algorithm(
        "ppo", config, model, solver, "cpu", state_dim=env.state.size
    )
    w_before = model.get_policy_params().copy()
    state = env.state
    for _ in range(config.total_iters):
        for _ in range(agent.rollout_iters):
            model.update_from_environment(env)
            decision = agent.act(state)
            action = env.sanitize_action(decision.action)
            old_state = state.copy()
            state, reward, terminated, _, info = env.step(action)
            agent.observe(
                train.AlgorithmTransition(
                    old_state_for_buffer=old_state,
                    reward=float(reward),
                    terminated=bool(terminated),
                    action_number=info["action"],
                    old_state=info["old_state"],
                    new_state=info["new_state"],
                    chosen_action_raw=decision.raw_action,
                    executed_action=action,
                ),
                decision.info,
            )
            if terminated:
                state, _ = env.reset()
        result = agent.update(state)
        metrics = result["metrics"]
    w_after = model.get_policy_params()
    check(np.all(np.isfinite(w_after)), "weights stay finite")
    check(not np.allclose(w_before, w_after), "PPO updated the MILP prize weights")
    check(metrics["buffer_size"] == agent.rollout_iters, "PPO consumed the rollout")
    eval_metrics = validation.evaluate(w_after)
    check(np.isfinite(eval_metrics["dvsp/val_reward"]),
          f"validation runs: {eval_metrics['dvsp/val_reward']:.2f} "
          f"({eval_metrics['dvsp/val_delta_greedy_pct']:+.1f}% vs greedy)")
    print("\nAll DVSP smoke checks passed.")


if __name__ == "__main__":
    main()
