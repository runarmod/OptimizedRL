"""Correctness and timing checks for the DAP integration (no wandb, a few minutes).

Usage: uv run python scripts/dap_smoke_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402
from src.config.config_loader import load_config  # noqa: E402
from src.dap import milp as dap_milp  # noqa: E402
from src.dap.evaluation import (  # noqa: E402
    evaluate,
    expert_policy,
    greedy_policy,
    milp_policy,
)
from src.dap.simulator import DAPSimulator  # noqa: E402
from src.models.dap_model import DAPModel  # noqa: E402
from src.solvers.exact_assortment import ExactAssortmentSolver  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok  {message}")


class Snapshot:
    def __init__(self, sim):
        self.state = np.zeros(1)
        self.features = sim.item_features()

    def item_features(self):
        return self.features


def lp(node, bounds=None):
    return linprog(node["c"], A_ub=node["A_ub"], b_ub=node["b_ub"], A_eq=node["A_eq"],
                   b_eq=node["b_eq"], bounds=bounds or node["bounds"], method="highs-ds")


def node_bounds(node, conds=(), fixed_x=None):
    bounds = list(node["bounds"])
    for var, op, value in conds:
        lb, ub = bounds[var]
        bounds[var] = (max(lb, value), ub) if op == ">=" else (lb, min(ub, value))
    if fixed_x is not None:
        for i, xi in enumerate(fixed_x):
            bounds[i] = (xi, xi)
    return bounds


def main():
    # --- environment and baselines ---------------------------------------
    greedy = evaluate(greedy_policy, range(1, 6))
    expert = evaluate(expert_policy, range(1, 6))
    check(expert.mean() > greedy.mean() > 0,
          f"expert {expert.mean():.1f} > greedy {greedy.mean():.1f} on episodes 1-5")

    sim = DAPSimulator()
    sim.reset(7)
    for _ in range(30):
        sim.step(greedy_policy(sim))
    snap = Snapshot(sim)
    rng = np.random.default_rng(3)
    theta = dap_milp.initial_theta(2, rng, pairwise=True)
    model = DAPModel(theta, n_pieces=2, pairwise=True)
    model.update_from_environment(snap)
    node = model.get_LP_formulation()
    n_vars = len(node["c"])

    # --- exact solver = MILP optimum -------------------------------------
    pool = ExactAssortmentSolver().solve(node)
    check(len(pool) == 4845, "exact solver enumerates all 4845 assortments")
    lb = [b[0] if b[0] is not None else -np.inf for b in node["bounds"]]
    ub = [b[1] if b[1] is not None else np.inf for b in node["bounds"]]
    res = milp(node["c"], constraints=[LinearConstraint(node["A_ub"], -np.inf, node["b_ub"]),
                                       LinearConstraint(node["A_eq"], node["b_eq"], node["b_eq"])],
               integrality=node["integer"], bounds=Bounds(lb, ub))
    check(np.isclose(res.fun, pool[0]["fun"], atol=1e-6),
          f"exact best Q {pool[0]['fun']:.4f} equals the pairwise MILP optimum {res.fun:.4f}")
    check(np.array_equal(milp_policy(theta, 2, True)(sim), pool[0]["x"][:20]),
          "deterministic policy is the exact MILP optimum")

    # --- gradients vs finite differences ----------------------------------
    direction = rng.normal(size=theta.size)
    eps = 1e-6

    def value_at(t, bounds):
        m = DAPModel(t, n_pieces=2, pairwise=True)
        m.update_from_environment(snap)
        return lp(m.get_LP_formulation(), bounds).fun

    def check_entry(entry, bounds, label):
        g = model.lagrange_gradient(entry["x"], snap.state, entry["eqlin"], entry["ineqlin"])
        fd = (value_at(theta + eps * direction, bounds) - value_at(theta, bounds)) / eps
        assert np.isclose(fd, g @ direction, rtol=1e-4, atol=1e-5), (label, fd, g @ direction)

    root = lp(node)
    check(not np.allclose(root.x[:20], np.round(root.x[:20]), atol=1e-6),
          "pairwise LP relaxation is fractional at this state")
    check_entry({"x": root.x, "eqlin": root.eqlin.marginals, "ineqlin": root.ineqlin.marginals},
                node["bounds"], "root")
    check(True, "root-LP gradient (duals) matches finite difference")
    for e in pool[:50:10]:
        check_entry(e, node_bounds(node, fixed_x=e["x"][:20]), "exact")
    check(True, "exact-solver candidate gradients match finite differences")

    # --- SCIP branch-and-bound pool ----------------------------------------
    config = load_config(PROJECT_ROOT / "config_dap.yaml")
    scip = train.build_solver(config)
    started = time.time()
    scip_pool = scip.solve(node) or []
    solve_ms = (time.time() - started) * 1000
    fractional = [e for e in scip_pool if not np.allclose(e["x"][:20], np.round(e["x"][:20]), atol=1e-6)]
    check(len(scip_pool) > 1 and fractional,
          f"SCIP tree gives {len(scip_pool)} candidates ({len(fractional)} pruned fractional) in {solve_ms:.0f} ms")
    for e in fractional[:3]:
        check_entry(e, node_bounds(node, conds=e["conds"]), "scip node")
    check(True, "SCIP node gradients match finite differences at the node's bounds")
    for e in scip_pool:
        action = model.complete_action(e)
        assert action.sum() == 4 and set(np.unique(action)) <= {0.0, 1.0}
        for var, op, value in e["conds"]:
            if var < 20:
                assert action[var] == (1 if op == ">=" and value >= 0.5 else action[var])
                assert action[var] == (0 if op == "<=" and value <= 0.5 else action[var])
    check(True, "every SCIP candidate completes (NNS-3) to a valid assortment respecting its branching")

    # --- bias-corrected node values ("completed") ---------------------------
    model.node_values = "completed"
    refined = model.refine_pool(scip_pool)
    model.node_values = "lp_bound"
    executed = np.array([model.complete_action(e) for e in refined])
    exact_q, _ = dap_milp.assortment_values(theta, 2, model.phi, executed, True)
    check(all(np.array_equal(executed[m], refined[m]["x"][:20]) for m in range(len(refined)))
          and np.allclose([e["fun"] for e in refined], exact_q),
          f"completed pool: {len(refined)} distinct assortments, each valued by its exact Q")
    for e in refined[:3]:
        check_entry(e, node_bounds(node, fixed_x=e["x"][:20]), "completed")
    check(True, "completed-candidate gradients match finite differences")

    # --- short PPO training with each config --------------------------------
    for name in ("config_dap.yaml", "config_dap_lpbound.yaml", "config_dap_exact.yaml",
                 "config_dap_linear.yaml"):
        config = load_config(PROJECT_ROOT / name)
        config = config.model_copy(update={
            "wandb_mode": "disabled",
            "dap": config.dap.model_copy(update={"eval_episodes": 3}),
            "ppo": config.ppo.model_copy(update={"rollout_iters": 40}),
        })
        model_run, env, validation = train.build_dap(config, PROJECT_ROOT)
        agent = train.build_training_algorithm("ppo", config, model_run, train.build_solver(config),
                                               "cpu", state_dim=env.state.size)
        before = model_run.get_policy_params().copy()
        state = env.state
        started = time.time()
        for _ in range(2):
            for _ in range(agent.rollout_iters):
                model_run.update_from_environment(env)
                decision = agent.act(state)
                action = env.sanitize_action(decision.action)
                old = state.copy()
                state, reward, terminated, _, info = env.step(action)
                agent.observe(train.AlgorithmTransition(
                    old_state_for_buffer=old, reward=float(reward), terminated=bool(terminated),
                    action_number=info["action"], old_state=info["old_state"],
                    new_state=info["new_state"], chosen_action_raw=decision.raw_action,
                    executed_action=action), decision.info)
                if terminated:
                    state, _ = env.reset()
            agent.update(state)
        per_step = (time.time() - started) / (2 * agent.rollout_iters)
        after = model_run.get_policy_params()
        metrics = validation.evaluate(after)
        check(np.all(np.isfinite(after)) and not np.allclose(before, after),
              f"{name}: PPO updates theta, {per_step * 1000:.0f} ms/step, "
              f"val reward {metrics['dap/val_reward']:.1f}")
    print("\nAll DAP smoke checks passed.")


if __name__ == "__main__":
    main()
