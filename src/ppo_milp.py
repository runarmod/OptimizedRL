"""PPO utilities for MILP-based policy optimization.

This module keeps PPO separate from the existing vanilla gradient actor so both
approaches can coexist in the same codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.actor import calc_actual_grad
from src.utils.policy import (
    categorical,
    logsumnp,
    nabla_log_pi_stable,
    naive_branch_sample,
    nn_branch_sample,
)


class RunningMeanStd:
    """Online mean/std tracker."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float):
        batch_mean = float(x)
        batch_var = 0.0
        batch_count = 1.0

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = max(new_var, 1e-12)
        self.count = tot_count


class RewardNormalizer:
    """Running discounted reward normalization."""

    def __init__(self, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8):
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.running_return = 0.0
        self.rms = RunningMeanStd()

    def normalize(self, reward: float, done: bool) -> float:
        self.running_return = self.running_return * self.gamma + float(reward)
        self.rms.update(self.running_return)
        std = np.sqrt(self.rms.var + self.epsilon)
        normalized = float(reward) / std
        normalized = float(np.clip(normalized, -self.clip, self.clip))
        if done:
            self.running_return = 0.0
        return normalized


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64, activation: str = "tanh"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        else:
            act = nn.Tanh
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    reward: float
    done: bool
    value: float
    logp: float
    score: np.ndarray


class PPOBuffer:
    def __init__(self):
        self.data: list[Transition] = []

    def add(self, transition: Transition):
        self.data.append(transition)

    def reset(self):
        self.data = []

    def __len__(self):
        return len(self.data)


class PPOMILP:
    """PPO optimizer over MILP parameters (aA, aB, b)."""

    def __init__(
        self,
        model,
        solver,
        state_dim: int,
        device: str = "cpu",
        beta: float = 0.05,
        sampled_grad: bool = False,
        nn_sample: bool = False,
        hidden_dim: int = 64,
        activation: str = "tanh",
        gamma: float = 0.99,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        use_clipped_value: bool = False,
        clip_param: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        opt_epochs: int = 10,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
        norm_reward: bool = True,
        clip_reward: float = 10.0,
    ):
        self.model = model
        self.solver = solver
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.desc_vars = self.model.get_desc_var_indices()
        self.beta = beta
        self.sampled_grad = sampled_grad
        self.nn_sample = nn_sample

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        self.reward_normalizer = None
        if norm_reward:
            self.reward_normalizer = RewardNormalizer(gamma=gamma, clip=clip_reward)

        self.value_net = ValueNetwork(
            obs_dim=state_dim, hidden_dim=hidden_dim, activation=activation
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)

        self.buffer = PPOBuffer()

    def _get_theta(self) -> np.ndarray:
        return np.concatenate(
            [
                self.model.aA.flatten(),
                self.model.aB.flatten(),
                self.model.b.flatten(),
            ]
        ).astype(np.float64)

    def _set_theta(self, theta: np.ndarray):
        idx = 0
        aA_size = self.model.aA.size
        aB_size = self.model.aB.size
        b_size = self.model.b.size

        self.model.aA = theta[idx : idx + aA_size].reshape(self.model.aA.shape)
        idx += aA_size
        self.model.aB = theta[idx : idx + aB_size].reshape(self.model.aB.shape)
        idx += aB_size
        self.model.b = theta[idx : idx + b_size].reshape(self.model.b.shape)

    def _value(self, state: np.ndarray) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            return float(self.value_net(state_t).squeeze().item())

    def act(self, new_state: np.ndarray):
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation()
        sol_pool = self.solver.solve(node)

        if sol_pool is None or len(sol_pool) == 0:
            return None, {"fathomed": False, "nab": None, "n_sols": 0, "t_nab": 0.0}

        obj_values = np.array([sol["fun"] for sol in sol_pool], dtype=float)
        logits = -self.beta * obj_values
        log_probs = logits - logsumnp(logits)
        probs = np.exp(log_probs)

        draw = int(categorical(probs))
        chosen_sol = sol_pool[draw]
        action = chosen_sol["x"][self.desc_vars]
        bounds = chosen_sol["bounds"]

        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action, bounds = nn_branch_sample(action, bounds)
            else:
                action, bounds = naive_branch_sample(action, bounds)

        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [np.asarray(sol["ineqlin"]) for sol in sol_pool]
        eq_margs = [np.asarray(sol["eqlin"]) for sol in sol_pool]
        lag_grads = [
            self.model.lagrange_gradient(a, new_state, eq_marg, ineq_marg)
            for a, ineq_marg, eq_marg in zip(actions, ineq_margs, eq_margs)
        ]

        if self.sampled_grad:
            node["bounds"] = bounds
            ineq, eq = calc_actual_grad(node)
            lag_grads[draw] = self.model.lagrange_gradient(action, new_state, eq, ineq)

        lag_grads = np.asarray(lag_grads, dtype=np.float64)
        score_full = nabla_log_pi_stable(
            lag_grads[draw], obj_values, lag_grads, self.beta
        )

        # The model lagrangian gradient is ordered as [dc, daA, daB, db].
        # PPO updates only [aA, aB, b], so drop dc components to match theta.
        c_size = int(np.asarray(self.model.c).size)
        score = np.asarray(score_full[c_size:], dtype=np.float64)

        info = {
            "fathomed": chosen_sol["fathomed"],
            "nab": score,
            "n_sols": len(sol_pool),
            "t_nab": 0.0,
            "ppo_score": score,
            "ppo_logp": float(log_probs[draw]),
            "ppo_value": self._value(new_state),
        }
        return action, info

    def update_buffers(
        self,
        reward: float,
        state: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        act_info: dict,
    ):
        del next_state
        if act_info.get("ppo_score") is None:
            return

        store_reward = float(reward)
        if self.reward_normalizer is not None:
            store_reward = self.reward_normalizer.normalize(store_reward, bool(done))

        self.buffer.add(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                reward=store_reward,
                done=bool(done),
                value=float(act_info["ppo_value"]),
                logp=float(act_info["ppo_logp"]),
                score=np.asarray(act_info["ppo_score"], dtype=np.float64).copy(),
            )
        )

    def _compute_returns_advantages(
        self, rewards, values, dones, last_value: float
    ) -> tuple[np.ndarray, np.ndarray]:
        returns = np.zeros_like(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards, dtype=np.float64)

        if self.use_gae:
            gae = 0.0
            vals_ext = np.concatenate([values, np.array([last_value], dtype=np.float64)])
            for t in reversed(range(len(rewards))):
                non_terminal = 1.0 - float(dones[t])
                delta = rewards[t] + self.gamma * vals_ext[t + 1] * non_terminal - vals_ext[t]
                gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
                advantages[t] = gae
                returns[t] = gae + vals_ext[t]
        else:
            running_return = last_value
            for t in reversed(range(len(rewards))):
                non_terminal = 1.0 - float(dones[t])
                running_return = rewards[t] + self.gamma * non_terminal * running_return
                returns[t] = running_return
                advantages[t] = running_return - values[t]

        return returns, advantages

    def _clip_grad(self, grad: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(grad))
        if self.max_grad_norm <= 0 or norm <= self.max_grad_norm:
            return grad
        return grad * (self.max_grad_norm / (norm + 1e-12))

    def train(
        self,
        iters: int = 0,
        sample: bool = False,
        num_samples: float = 1.0,
        last_state: np.ndarray | None = None,
        last_done: bool = False,
    ):
        del iters, sample, num_samples

        if len(self.buffer) == 0:
            raise Exception("PPO buffer is empty")

        states = np.stack([t.state for t in self.buffer.data]).astype(np.float32)
        rewards = np.array([t.reward for t in self.buffer.data], dtype=np.float64)
        dones = np.array([t.done for t in self.buffer.data], dtype=np.bool_)
        values = np.array([t.value for t in self.buffer.data], dtype=np.float64)
        logp_old = np.array([t.logp for t in self.buffer.data], dtype=np.float64)
        scores = np.stack([t.score for t in self.buffer.data]).astype(np.float64)

        last_value = 0.0
        if (last_state is not None) and (not last_done):
            last_value = self._value(last_state)

        returns, advantages = self._compute_returns_advantages(
            rewards, values, dones, last_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        theta_old = self._get_theta().copy()
        theta = theta_old.copy()

        # --- diagnostic ---
        print(
            f"[PPO train] buf={len(rewards)} "
            f"| rew mean={rewards.mean():.4f} std={rewards.std():.4f} "
            f"| adv mean={advantages.mean():.4f} std={advantages.std():.4f} "
            f"| logp mean={logp_old.mean():.4f} "
            f"| score_norm mean={float(np.linalg.norm(scores, axis=1).mean()):.4f} "
            f"| theta_norm={float(np.linalg.norm(theta)):.4f}"
        )
        # ------------------

        if scores.shape[1] != theta.shape[0]:
            raise ValueError(
                f"PPO score/theta mismatch: score_dim={scores.shape[1]} theta_dim={theta.shape[0]}"
            )

        n_samples = states.shape[0]
        mini_batch_size = min(self.mini_batch_size, n_samples)

        policy_loss_acc = []
        value_loss_acc = []
        approx_kl_acc = []
        entropy_acc = []
        grad_norm_acc = []

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values_old_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)

        for _ in range(self.opt_epochs):
            perm = np.random.permutation(n_samples)
            stop_actor = False

            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                batch_idx = perm[start:end]
                adv_b = advantages[batch_idx]
                score_b = scores[batch_idx]

                delta_theta = theta - theta_old
                log_ratio = score_b @ delta_theta
                log_ratio = np.clip(log_ratio, -20.0, 20.0)
                ratio = np.exp(log_ratio)

                unclipped_obj = ratio * adv_b
                clipped_ratio = np.clip(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                clipped_obj = clipped_ratio * adv_b

                use_unclipped = unclipped_obj <= clipped_obj
                grad_terms = (ratio * adv_b)[:, None] * score_b
                actor_grad = grad_terms[use_unclipped].sum(axis=0) / max(1, len(batch_idx))

                actor_grad = self._clip_grad(actor_grad)
                theta = theta + self.actor_lr * actor_grad
                self._set_theta(theta)

                approx_kl = float(np.mean(-log_ratio))
                # --- diagnostic ---
                print(
                    f"  [batch] use_unclipped={use_unclipped.sum()}/{len(batch_idx)} "
                    f"| ratio mean={ratio.mean():.4f} "
                    f"| approx_kl={approx_kl:.6f} "
                    f"| actor_grad_norm={float(np.linalg.norm(actor_grad)):.6f} "
                    f"| delta_theta_norm={float(np.linalg.norm(theta - theta_old)):.6f}"
                )
                # ------------------
                if self.target_kl > 0 and abs(approx_kl) > 1.5 * self.target_kl:
                    print(f"  [PPO] early stop: approx_kl={approx_kl:.6f} > 1.5*target_kl={1.5*self.target_kl:.6f}")
                    stop_actor = True

                with torch.no_grad():
                    entropy = float(-np.mean(logp_old[batch_idx]))

                policy_loss = -float(np.mean(np.minimum(unclipped_obj, clipped_obj)))
                policy_loss_acc.append(policy_loss)
                approx_kl_acc.append(approx_kl)
                entropy_acc.append(entropy)
                grad_norm_acc.append(float(np.linalg.norm(actor_grad)))

                b_states = states_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_values_old = values_old_t[batch_idx]

                values_pred = self.value_net(b_states).squeeze(-1)
                if self.use_clipped_value:
                    v_clipped = b_values_old + (values_pred - b_values_old).clamp(
                        -self.clip_param, self.clip_param
                    )
                    loss_v1 = (values_pred - b_returns).pow(2)
                    loss_v2 = (v_clipped - b_returns).pow(2)
                    value_loss = 0.5 * torch.max(loss_v1, loss_v2).mean()
                else:
                    value_loss = 0.5 * (values_pred - b_returns).pow(2).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                value_loss_acc.append(float(value_loss.item()))

                if stop_actor:
                    break

            if stop_actor:
                break

        self.buffer.reset()

        return {
            "policy_loss": float(np.mean(policy_loss_acc)) if policy_loss_acc else 0.0,
            "value_loss": float(np.mean(value_loss_acc)) if value_loss_acc else 0.0,
            "entropy_loss": float(np.mean(entropy_acc)) if entropy_acc else 0.0,
            "approx_kl": float(np.mean(approx_kl_acc)) if approx_kl_acc else 0.0,
            "policy_grad_norm": float(np.mean(grad_norm_acc)) if grad_norm_acc else 0.0,
        }
