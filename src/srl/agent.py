"""SRL agent for DAP: experience collection, actor and critic updates.

Port of `SRL_dynamic` (scripts/DAP/03_SRL.jl) and `generate_episode`
(scripts/DAP/utils/utils.jl) of the SRL codebase.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn.functional import huber_loss

from src.gym_envs.dap_env import N_ITEM_FEATURES, DynamicAssortmentEnvironment
from src.srl.buffer import ReplayBuffer, Transition, with_returns
from src.srl.co_layer import top_k, top_k_numpy
from src.srl.config import SrlConfig
from src.srl.fenchel_young import perturbed_fenchel_young_loss
from src.srl.networks import Actor, FutureCritic, RewardCritic
from src.srl.target import softmax_target

HUBER_DELTA = 1.0

# Seed of the batch order of the future critic. The SRL codebase builds its
# DataLoader with MersenneTwister(0) anew every episode.
FUTURE_CRITIC_SHUFFLE_SEED = 0


def _as_tensor(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


class SrlAgent:
    def __init__(
        self,
        config: SrlConfig,
        actor: Actor,
        reward_critic: RewardCritic,
        future_critic: FutureCritic,
    ):
        self.config = config
        self.actor = actor
        self.reward_critic = reward_critic
        self.future_critic = future_critic
        lr = config.lr.start
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        self.reward_critic_opt = torch.optim.Adam(reward_critic.parameters(), lr=lr)
        self.future_critic_opt = torch.optim.Adam(future_critic.parameters(), lr=lr)
        self.buffer = ReplayBuffer(config.buffer_size, np.random.default_rng(config.seed))
        self.fy_rng = np.random.default_rng([config.seed, 1])
        self.completed_episodes = 0
        self._linear_actor: tuple[np.ndarray, float] | None = None

    @property
    def n_items(self) -> int:
        return self.config.env.n_items

    @property
    def assortment_size(self) -> int:
        return self.config.env.assortment_size

    def _scheduled(self, name: str) -> float:
        schedule = getattr(self.config, name)
        return schedule.value(self.completed_episodes, self.config.episodes)

    @property
    def sigma_f(self) -> float:
        return self._scheduled("sigma_f")

    @property
    def sigma_b(self) -> float:
        return self._scheduled("sigma_b")

    @property
    def temperature(self) -> float:
        return self._scheduled("temperature")

    @property
    def lr(self) -> float:
        return self._scheduled("lr")

    def scores(self, observation: np.ndarray) -> np.ndarray:
        """Actor scores theta of the items in an environment observation."""
        if self._linear_actor is None:
            self._linear_actor = self.actor.linear_scores()
        w, b = self._linear_actor
        features = np.asarray(observation, dtype=float).reshape(self.n_items, -1)
        return features @ w + b

    def actor_updated(self) -> None:
        """Call after changing the actor's weights, so `scores` uses them."""
        self._linear_actor = None

    def best_assortment(self, observation: np.ndarray) -> np.ndarray:
        """Deterministic policy: top-K of the unperturbed scores."""
        return top_k_numpy(self.scores(observation), self.assortment_size)

    def collect_episode(
        self, env: DynamicAssortmentEnvironment, episode_seed: int
    ) -> list[Transition]:
        """One episode with exploration noise, with returns (`generate_episode`)."""
        observation, _ = env.reset(seed=episode_seed)
        noise_std = math.sqrt(self.sigma_f)
        transitions = []
        terminated = False
        while not terminated:
            # The SRL codebase draws the noise with MersenneTwister(seed * step).
            noise_rng = np.random.default_rng(episode_seed * env.sim.current_step)
            eta = self.scores(observation) + noise_std * noise_rng.standard_normal(
                self.n_items
            )
            assortment = top_k_numpy(eta, self.assortment_size)
            features = observation.reshape(self.n_items, N_ITEM_FEATURES)
            observation, revenue, terminated, _, _ = env.step(assortment)
            transitions.append(
                Transition(
                    features=features.astype(np.float32),
                    assortment=assortment.astype(np.float32),
                    revenue=float(revenue),
                )
            )
        return with_returns(transitions, self.config.gamma)

    def critic_values(
        self, features: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Q-values (batch, n_candidates) of candidate assortments
        (batch, n_candidates, n_items) in states (batch, n_items, F)."""
        batch, n_candidates, n_items = candidates.shape
        flat_features = (
            features.unsqueeze(1)
            .expand(batch, n_candidates, n_items, features.shape[-1])
            .reshape(batch * n_candidates, n_items, -1)
        )
        flat_candidates = candidates.reshape(batch * n_candidates, n_items)
        values = self.reward_critic(flat_features, flat_candidates)
        if self.config.dynamic:
            values = values + self.future_critic(flat_features, flat_candidates)
        return values.reshape(batch, n_candidates)

    def target_actions(
        self, features: torch.Tensor, perturbation_seed: int
    ) -> torch.Tensor:
        """Softmax target actions (batch, n_items) of a batch of states.

        The candidates are the unperturbed action and the actions of
        `target_samples` perturbed score vectors. As in the SRL codebase
        (MersenneTwister(e * i)), all states of a batch share the noise draw.
        """
        cfg = self.config
        k = self.assortment_size
        with torch.no_grad():
            theta = self.actor(features)
            noise_rng = np.random.default_rng(perturbation_seed)
            noise = _as_tensor(noise_rng.standard_normal((cfg.target_samples, self.n_items)))
            eta = theta.unsqueeze(1) + math.sqrt(self.sigma_b) * noise
            candidates = torch.cat((top_k(theta, k).unsqueeze(1), top_k(eta, k)), dim=1)
            values = self.critic_values(features, candidates)
            return softmax_target(candidates, values, self.temperature)

    def optimizer_step(
        self, optimizer: torch.optim.Optimizer, module: nn.Module, loss: torch.Tensor
    ) -> None:
        """Backpropagate loss, clip the gradients elementwise and step."""
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(module.parameters(), self.config.grad_clip)
        optimizer.step()

    def update(self, transitions: list[Transition]) -> dict[str, float]:
        """Training after an episode: `iterations` actor and reward-critic updates
        from the replay buffer, then the future critic on the new episode."""
        cfg = self.config
        episode = self.completed_episodes + 1
        self.buffer.add(transitions)
        batches = [self.buffer.sample(cfg.batch_size) for _ in range(cfg.iterations)]

        actor_losses, reward_losses = [], []
        for i, batch in enumerate(batches, start=1):
            features = _as_tensor([t.features for t in batch])
            assortments = _as_tensor([t.assortment for t in batch])
            revenues = _as_tensor([t.revenue for t in batch])

            targets = self.target_actions(features, episode * i)

            fy_noise = _as_tensor(
                self.fy_rng.standard_normal((len(batch), cfg.fy_samples, self.n_items))
            )
            actor_loss = perturbed_fenchel_young_loss(
                self.actor(features), targets, self.assortment_size, cfg.fy_epsilon, fy_noise
            ).sum()
            self.optimizer_step(self.actor_opt, self.actor, actor_loss)
            self.actor_updated()

            reward_loss = huber_loss(
                self.reward_critic(features, assortments), revenues, delta=HUBER_DELTA
            )
            self.optimizer_step(self.reward_critic_opt, self.reward_critic, reward_loss)

            actor_losses.append(actor_loss.item())
            reward_losses.append(reward_loss.item())

        metrics = {
            "actor_loss": float(np.mean(actor_losses)),
            "reward_critic_loss": float(np.mean(reward_losses)),
        }
        if cfg.dynamic:
            metrics["future_critic_loss"] = self._train_future_critic(transitions)
        return metrics

    def _train_future_critic(self, transitions: list[Transition]) -> float:
        """Huber regression of the return after each step, ret_t - rev_t, on
        shuffled batches of the episode; incomplete batches are skipped."""
        cfg = self.config
        features = _as_tensor([t.features for t in transitions])
        assortments = _as_tensor([t.assortment for t in transitions])
        targets = _as_tensor([t.ret - t.revenue for t in transitions])
        shuffle_rng = np.random.default_rng(FUTURE_CRITIC_SHUFFLE_SEED)
        n_batches = len(transitions) // cfg.batch_size
        losses = []
        for _ in range(cfg.future_critic_passes):
            order = torch.as_tensor(shuffle_rng.permutation(len(transitions)))
            for b in range(n_batches):
                idx = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                loss = huber_loss(
                    self.future_critic(features[idx], assortments[idx]),
                    targets[idx],
                    delta=HUBER_DELTA,
                )
                self.optimizer_step(self.future_critic_opt, self.future_critic, loss)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def end_episode(self) -> None:
        """Advance the noise, temperature and learning-rate schedules."""
        self.completed_episodes += 1
        for optimizer in (self.actor_opt, self.reward_critic_opt, self.future_critic_opt):
            for group in optimizer.param_groups:
                group["lr"] = self.lr
