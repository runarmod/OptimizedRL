"""Correctness and timing checks for the DAP integration (no wandb, ~2 minutes).

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
from src.solvers.enumeration import BranchEnumerationSolver  # noqa: E402
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


def main():
    # --- environment and baselines ---------------------------------------
    greedy = evaluate(greedy_policy, range(1, 6))
    expert = evaluate(expert_policy, range(1, 6))
    check(expert.mean() > greedy.mean() > 0,
          f"expert {expert.mean():.1f} > greedy {greedy.mean():.1f} on episodes 1-5")

    # A decision state from the middle of an episode.
    sim = DAPSimulator()
    sim.reset(7)
    for _ in range(30):
        sim.step(greedy_policy(sim))
    snap = Snapshot(sim)
    rng = np.random.default_rng(0)
    theta = dap_milp.initial_theta(2, rng)
    model = DAPModel(theta, n_pieces=2)
    model.update_from_environment(snap)
    node = model.get_LP_formulation()

    # --- exact solver = MILP optimum -------------------------------------
    pool = ExactAssortmentSolver().solve(node)
    check(len(pool) == 4845, "exact solver enumerates all 4845 assortments")
    res = milp(node["c"], constraints=[LinearConstraint(node["A_ub"], -np.inf, node["b_ub"]),
                                       LinearConstraint(node["A_eq"], node["b_eq"], node["b_eq"])],
               integrality=node["integer"], bounds=Bounds([0] * 20 + [-np.inf], [1] * 20 + [np.inf]))
    check(np.isclose(res.fun, pool[0]["fun"], atol=1e-6),
          f"exact best Q {pool[0]['fun']:.4f} equals MILP optimum {res.fun:.4f}")

    # --- gradients vs finite differences ----------------------------------
    direction = rng.normal(size=theta.size)
    eps = 1e-6

    def value_at(t, x=None):
        m = DAPModel(t, n_pieces=2)
        m.update_from_environment(snap)
        n = m.get_LP_formulation()
        if x is None:
            return lp(n).fun
        bounds = [(xi, xi) for xi in x[:20]] + [(None, None)]
        return lp(n, bounds).fun

    root = lp(node)
    grad = model.lagrange_gradient(root.x, snap.state, root.eqlin.marginals, root.ineqlin.marginals)
    fd = (value_at(theta + eps * direction) - root.fun) / eps
    check(np.isclose(fd, grad @ direction, rtol=1e-4, atol=1e-5),
          f"root-LP gradient (duals) matches finite difference ({fd:.4f})")
    grads = model.lagrange_gradient_batch(np.array([e["x"] for e in pool[:50]]), snap.state,
                                          [e["eqlin"] for e in pool[:50]],
                                          [e["ineqlin"] for e in pool[:50]])
    for e, g in list(zip(pool[:50], grads))[::10]:
        fd = (value_at(theta + eps * direction, e["x"]) - value_at(theta, e["x"])) / eps
        assert np.isclose(fd, g @ direction, rtol=1e-4, atol=1e-5), (fd, g @ direction)
    check(True, "exact-solver candidate gradients match finite differences")

    # --- B&B tree and NNS-k completion --------------------------------------
    bnb_pool = BranchEnumerationSolver(max_pool_size=16, max_depth=2, integral_only=False).solve(node)
    check(len(bnb_pool) > 1, f"explicit B&B tree gives {len(bnb_pool)} candidates")
    for entry in bnb_pool:
        action = model.complete_action(entry)
        assert action.sum() == 4 and set(np.unique(action)) <= {0.0, 1.0}
    check(True, "every candidate completes to a valid assortment")
    fractional = {"x": np.r_[np.full(20, 0.2), 0.0], "fixings": ((3, 1), (5, 0))}
    picks = [model.complete_action(fractional) for _ in range(20)]
    check(all(p[3] == 1 and p[5] == 0 and p.sum() == 4 for p in picks),
          "NNS-k respects branching fixings")

    # --- deterministic policy == argmin over all assortments ----------------
    det = milp_policy(theta, 2)(sim)
    check(np.array_equal(det, pool[0]["x"][:20]), "deterministic policy is the exact MILP optimum")

    # --- short PPO training with each config --------------------------------
    for name in ("config_dap.yaml", "config_dap_exact.yaml", "config_dap_linear.yaml"):
        config = load_config(PROJECT_ROOT / name)
        config = config.model_copy(update={
            "wandb_mode": "disabled",
            "dap": config.dap.model_copy(update={"eval_episodes": 3}),
            "ppo": config.ppo.model_copy(update={"rollout_iters": 40}),
        })
        model, env, validation = train.build_dap(config, PROJECT_ROOT)
        agent = train.build_training_algorithm("ppo", config, model, train.build_solver(config),
                                               "cpu", state_dim=env.state.size)
        before = model.get_policy_params().copy()
        state = env.state
        started = time.time()
        for _ in range(2):
            for _ in range(agent.rollout_iters):
                model.update_from_environment(env)
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
        after = model.get_policy_params()
        metrics = validation.evaluate(after)
        check(np.all(np.isfinite(after)) and not np.allclose(before, after),
              f"{name}: PPO updates theta, {per_step * 1000:.0f} ms/step, "
              f"val reward {metrics['dap/val_reward']:.1f}")
    print("\nAll DAP smoke checks passed.")


if __name__ == "__main__":
    main()
