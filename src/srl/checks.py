"""Checks that the parts of the SRL port learn, before full SRL runs count.

    uv run python -m src.srl.checks sil      # actor + Fenchel-Young loss on expert actions
    uv run python -m src.srl.checks critic   # reward critic against the true expected revenue

`sil` ports `SIL_dynamic!` (scripts/DAP/01_SIL.jl of the SRL codebase): the
actor imitates the expert with the Fenchel-Young loss, without critics. The
paper reports SIL at +11.7 % (train) and +7.3 % (test) over greedy.

`critic` trains the reward critic on experience of the initial actor with the
first episode's exploration noise, and measures how well its values rank
assortments by the environment's true expected revenue on held-out states.
"""

import argparse
import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn.functional import huber_loss

from src.gym_envs import dap_env
from src.srl.agent import HUBER_DELTA, SrlAgent
from src.srl.checkpoint import build_critics
from src.srl.co_layer import top_k, top_k_numpy
from src.srl.config import SrlConfig
from src.srl.evaluate import make_env, run_policy
from src.srl.fenchel_young import perturbed_fenchel_young_loss
from src.srl.train import initial_actor, validate

TRAIN_SEEDS = range(1, 101)
TEST_SEEDS = range(2001, 2101)


def new_agent(config: SrlConfig) -> SrlAgent:
    torch.set_num_threads(config.torch_threads)
    generator = torch.Generator().manual_seed(config.seed)
    actor, draw, reward = initial_actor(config, make_env(config.env), generator)
    print(f"Initial actor: draw {draw}, mean revenue {reward:.1f}")
    return SrlAgent(config, actor, *build_critics(config, generator))


def report(config: SrlConfig, label: str, policy) -> None:
    """Mean revenue of a policy on the paper's train and test episodes, next to
    greedy and expert, as gain over greedy and share of the expert's gain."""
    env = make_env(config.env)
    steps = config.env.max_steps
    print(f"\n{'policy':<8} {'split':<6} {'mean rev':>9} {'gain_greedy %':>14} {'expert_gap %':>13}")
    for split, seeds in (("train", TRAIN_SEEDS), ("test", TEST_SEEDS)):
        means = {
            name: np.mean([row["ep_reward"] for row in run_policy(env, act, seeds, steps)])
            for name, act in (
                ("greedy", lambda _: dap_env.greedy_assortment(env.sim)),
                ("expert", lambda _: dap_env.expert_assortment(env.sim)),
                (label, policy),
            )
        }
        for name, mean in means.items():
            gain = 100 * (mean - means["greedy"]) / means["greedy"]
            gap = 100 * (mean - means["greedy"]) / (means["expert"] - means["greedy"])
            print(f"{name:<8} {split:<6} {mean:>9.1f} {gain:>+14.1f} {gap:>+13.1f}")


def expert_dataset(config: SrlConfig) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Expert (features, assortment) pairs of every training episode
    (`expert_policy(...; create_dataset=true)`)."""
    env = make_env(config.env)
    episodes = []
    for seed in TRAIN_SEEDS:
        observation, _ = env.reset(seed=seed)
        pairs = []
        terminated = False
        while not terminated:
            assortment = dap_env.expert_assortment(env.sim)
            pairs.append(
                (observation.reshape(config.env.n_items, -1), assortment.astype(np.float32))
            )
            observation, _, terminated, _, _ = env.step(assortment)
        episodes.append(pairs)
    return episodes


def check_sil(config: SrlConfig, epochs: int, lr: float) -> None:
    agent = new_agent(config)
    data = expert_dataset(config)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    validation_env = make_env(config.env)
    k = config.env.assortment_size
    best_state, best_score, best_epoch = None, None, None
    for epoch in range(1, epochs + 1):
        for i, episode in enumerate(data):
            # Validate every fourth episode; only the second half of training may
            # give the best model, to avoid a "lucky shot" early on.
            if i % 4 == 0:
                score = float(np.mean(list(validate(agent, validation_env, config).values())))
                if epoch > epochs / 2 and (best_score is None or score >= best_score):
                    best_score, best_epoch = score, epoch
                    best_state = {n: p.clone() for n, p in agent.actor.state_dict().items()}
            for features, assortment in episode:
                noise = torch.as_tensor(
                    agent.fy_rng.standard_normal((1, config.fy_samples, config.env.n_items)),
                    dtype=torch.float32,
                )
                theta = agent.actor(torch.as_tensor(features, dtype=torch.float32)[None])
                loss = perturbed_fenchel_young_loss(
                    theta, torch.as_tensor(assortment)[None], k, config.fy_epsilon, noise
                ).sum()
                agent._step(optimizer, agent.actor, loss)
            agent._linear_actor = None
        print(f"epoch {epoch}: last validation score {score:.1f}")
    agent.actor.load_state_dict(best_state)
    agent._linear_actor = None
    print(f"Best validation score {best_score:.1f} in epoch {best_epoch}")
    report(config, "SIL", agent.best_assortment)


def exploration_data(agent: SrlAgent, env, seeds) -> list[dict]:
    """States visited with the first episode's exploration noise, with what the
    true expected revenue of any assortment needs: utilities and prices."""
    k = agent.assortment_size
    noise_std = math.sqrt(agent.sigma_f)
    rng = np.random.default_rng(0)
    states = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = False
        while not terminated:
            eta = agent.scores(observation) + noise_std * rng.standard_normal(agent.n_items)
            assortment = top_k_numpy(eta, k)
            states.append(
                {
                    "features": observation.reshape(agent.n_items, -1).astype(np.float32),
                    "assortment": assortment.astype(np.float32),
                    "utilities": env.sim.customer_utilities().copy(),
                    "prices": env.sim.prices.copy(),
                }
            )
            observation, revenue, terminated, _, _ = env.step(assortment)
            states[-1]["revenue"] = revenue
    return states


def expected_revenue(state: dict, assortment: np.ndarray) -> float:
    probs = dap_env.DynamicAssortmentSimulator.choice_probabilities(
        state["utilities"], assortment
    )
    return float(probs[:-1] @ state["prices"])


def critic_ranking(agent: SrlAgent, states: list[dict], n_candidates: int, seed: int) -> tuple[float, float]:
    """Spearman correlation of reward-critic values with the true expected
    revenue: pooled over all (state, candidate) pairs, and the mean within a
    state over its candidates, the ranking the target action relies on."""
    rng = np.random.default_rng(seed)
    noise_std = math.sqrt(agent.sigma_b)
    pooled_values, pooled_truth, within = [], [], []
    with torch.no_grad():
        for state in states:
            theta = torch.as_tensor(agent.scores(state["features"]), dtype=torch.float32)
            eta = theta + noise_std * torch.as_tensor(
                rng.standard_normal((n_candidates, agent.n_items)), dtype=torch.float32
            )
            candidates = top_k(eta, agent.assortment_size)
            features = torch.as_tensor(state["features"]).expand(n_candidates, -1, -1)
            values = agent.reward_critic(features, candidates).numpy()
            truth = np.array([expected_revenue(state, c.numpy()) for c in candidates])
            pooled_values.extend(values)
            pooled_truth.extend(truth)
            if np.ptp(truth) > 0 and np.ptp(values) > 0:
                within.append(spearmanr(values, truth).statistic)
    return float(spearmanr(pooled_values, pooled_truth).statistic), float(np.mean(within))


def check_critic(config: SrlConfig, iterations: int, n_candidates: int) -> None:
    agent = new_agent(config)
    env = make_env(config.env)
    train_states = exploration_data(agent, env, TRAIN_SEEDS)
    held_out = exploration_data(agent, env, range(1001, 1031))
    rng = np.random.default_rng(config.seed)

    def loss_on(states: list[dict]) -> float:
        with torch.no_grad():
            features = torch.as_tensor(np.stack([s["features"] for s in states]))
            assortments = torch.as_tensor(np.stack([s["assortment"] for s in states]))
            revenues = torch.as_tensor([s["revenue"] for s in states], dtype=torch.float32)
            return huber_loss(agent.reward_critic(features, assortments), revenues, delta=HUBER_DELTA).item()

    print(f"{len(train_states)} training states, {len(held_out)} held-out states")
    print(f"before: Huber loss {loss_on(held_out):.2f}, Spearman pooled/within "
          "{:.3f} / {:.3f}".format(*critic_ranking(agent, held_out, n_candidates, 1)))
    optimizer = torch.optim.Adam(agent.reward_critic.parameters(), lr=config.lr.start)
    for it in range(1, iterations + 1):
        batch = [train_states[i] for i in rng.integers(len(train_states), size=config.batch_size)]
        loss = huber_loss(
            agent.reward_critic(
                torch.as_tensor(np.stack([s["features"] for s in batch])),
                torch.as_tensor(np.stack([s["assortment"] for s in batch])),
            ),
            torch.as_tensor([s["revenue"] for s in batch], dtype=torch.float32),
            delta=HUBER_DELTA,
        )
        agent._step(optimizer, agent.reward_critic, loss)
        if it % (iterations // 5) == 0:
            print(f"iteration {it}: held-out Huber loss {loss_on(held_out):.2f}")
    print("after: Spearman pooled/within {:.3f} / {:.3f}".format(
        *critic_ranking(agent, held_out, n_candidates, 1)))


def main():
    parser = argparse.ArgumentParser(description="Checks of the SRL port.")
    sub = parser.add_subparsers(dest="check", required=True)
    sil = sub.add_parser("sil", help="Structured imitation of the expert.")
    sil.add_argument("--epochs", type=int, default=8)
    sil.add_argument("--lr", type=float, default=1e-4)
    critic = sub.add_parser("critic", help="Reward critic against the true expected revenue.")
    critic.add_argument("--iterations", type=int, default=20_000)
    critic.add_argument("--candidates", type=int, default=40)
    for p in (sil, critic):
        p.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    config = SrlConfig(seed=args.seed)
    if args.check == "sil":
        check_sil(config, args.epochs, args.lr)
    else:
        check_critic(config, args.iterations, args.candidates)


if __name__ == "__main__":
    main()
