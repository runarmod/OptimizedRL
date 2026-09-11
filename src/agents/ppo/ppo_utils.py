"""PPO-MILP utilities.

This module is intentionally MILP-focused and does not depend on MPC code.
It assumes that each decision step can provide a pool of MILP candidates,
where the policy is a categorical distribution over candidate objective values.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.utils.policy import knn_branch_sample, naive_branch_sample


def _as_state_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    return torch.as_tensor(state_arr, dtype=torch.float32, device=device)


def _stable_categorical_logits(
    obj_t: torch.Tensor,
    beta: float,
    normalize_obj_values: bool,
    obj_norm_eps: float,
) -> torch.Tensor:
    # Keep objective values finite before normalization and logit construction.
    obj_t = torch.nan_to_num(obj_t, nan=0.0, posinf=1e6, neginf=-1e6)

    if normalize_obj_values and obj_t.numel() > 1:
        mean = obj_t.mean()
        std = obj_t.std(unbiased=False)
        if torch.isfinite(std) and std > 0:
            obj_t = (obj_t - mean) / (std + obj_norm_eps)
        else:
            obj_t = obj_t - mean

    logits = -float(beta) * obj_t
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    # Shift for numeric stability; this does not change the categorical distribution.
    if logits.numel() > 1:
        logits = logits - logits.max()

    if not torch.isfinite(logits).all():
        logits = torch.zeros_like(logits)

    return logits


class ValueNet(nn.Module):
    """Simple value network used by PPO for scalar state value estimation."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


@dataclass
class PPOStep:
    state: np.ndarray
    obj_vals: np.ndarray
    theta_grads: np.ndarray
    theta_ref: np.ndarray
    action_idx: int
    reward: float
    done: bool
    old_logp: float
    value: float
    chosen_action_raw: np.ndarray | None = None
    executed_action: np.ndarray | None = None


class PPORolloutBuffer:
    """Rollout buffer for PPO over MILP candidate pools."""

    def __init__(self):
        self.steps: list[PPOStep] = []

    def clear(self) -> None:
        self.steps.clear()

    def add(self, step: PPOStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def sampler(self, mini_batch_size: int, drop_last: bool = True):
        total = len(self.steps)
        indices = np.random.permutation(np.arange(total))
        n_full = total // mini_batch_size
        for i in range(n_full):
            batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            yield batch_idx, [self.steps[j] for j in batch_idx]
        if not drop_last and total % mini_batch_size != 0:
            rem = indices[n_full * mini_batch_size :]
            yield rem, [self.steps[j] for j in rem]


class PPOBuffer(PPORolloutBuffer):
    """Compatibility alias matching naming from PPO-MPC utilities."""


def compute_returns_and_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and discounted returns."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    next_adv = 0.0
    next_value = float(last_value)
    for t in reversed(range(n)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        next_adv = delta + gamma * gae_lambda * mask * next_adv
        adv[t] = next_adv
        returns[t] = adv[t] + values[t]
        next_value = values[t]

    return returns, adv


class PPOMILPAgent:
    """PPO agent for MILP candidate-set action selection.

    Policy parameterization:
    - Learned MILP parameters theta = (aA, aB, b).
    - Action probabilities use a fixed-temperature softmax over objective values.
    - PPO updates use a first-order linearization of objective values around theta_old.

    The existing MILP model and solver are reused directly.
    """

    def __init__(
        self,
        model,
        solver,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        entropy_coef: float = 0.001,
        value_coef: float = 0.5,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        policy_beta: float = 1.0,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        target_kl: float | None = None,
        normalize_adv: bool = True,
        normalize_obj_values: bool = True,
        obj_norm_eps: float = 1e-8,
        minimize_env_reward: bool = True,
        normalize_rewards: bool = True,
        reward_norm_eps: float = 1e-8,
        reward_clip: float | None = None,
        nn_sample: bool = True,
        device: str = "cpu",
    ):
        self.model = model
        self.solver = solver
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lr_policy = float(lr_policy)
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = None if target_kl is None else float(target_kl)
        self.normalize_adv = normalize_adv
        self.normalize_obj_values = bool(normalize_obj_values)
        self.obj_norm_eps = float(obj_norm_eps)
        self.minimize_env_reward = bool(minimize_env_reward)
        self.normalize_rewards = bool(normalize_rewards)
        self.reward_norm_eps = float(reward_norm_eps)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.policy_beta = float(policy_beta)
        self.nn_sample = bool(nn_sample)
        self.device = torch.device(device)

        aA0, aB0, b0 = self._get_model_param_arrays()
        theta0 = self._flatten_theta(aA0, aB0, b0)
        self.theta_dim = int(theta0.size)
        self.theta = nn.Parameter(
            torch.as_tensor(theta0, dtype=torch.float32, device=self.device)
        )
        self.value_net = ValueNet(state_dim=state_dim).to(self.device)
        self.policy_opt = torch.optim.Adam([self.theta], lr=lr_policy)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)

    def _get_model_param_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = None
        if hasattr(self.model, "get_params"):
            try:
                params = self.model.get_params()
            except Exception:
                params = None

        if isinstance(params, dict) and all(k in params for k in ("aA", "aB", "b")):
            return (
                np.asarray(params["aA"], dtype=np.float32),
                np.asarray(params["aB"], dtype=np.float32),
                np.asarray(params["b"], dtype=np.float32),
            )

        return (
            np.asarray(self.model.aA, dtype=np.float32),
            np.asarray(self.model.aB, dtype=np.float32),
            np.asarray(self.model.b, dtype=np.float32),
        )

    @staticmethod
    def _flatten_theta(aA: np.ndarray, aB: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(aA, dtype=np.float32).reshape(-1),
                np.asarray(aB, dtype=np.float32).reshape(-1),
                np.asarray(b, dtype=np.float32).reshape(-1),
            ]
        )

    def _theta_to_model(self, theta_vec: np.ndarray) -> None:
        aA_shape = self.model.aA.shape
        aB_shape = self.model.aB.shape
        b_shape = self.model.b.shape

        idx = 0
        aA_size = int(np.prod(aA_shape))
        aB_size = int(np.prod(aB_shape))
        b_size = int(np.prod(b_shape))

        self.model.aA = theta_vec[idx : idx + aA_size].reshape(aA_shape).astype(float)
        idx += aA_size
        self.model.aB = theta_vec[idx : idx + aB_size].reshape(aB_shape).astype(float)
        idx += aB_size
        self.model.b = theta_vec[idx : idx + b_size].reshape(b_shape).astype(float)

    def _sync_model_params_from_theta(self) -> None:
        theta_np = self.theta.detach().cpu().numpy().astype(np.float32)
        self._theta_to_model(theta_np)

    def to(self, device):
        self.device = torch.device(device)
        self.theta.data = self.theta.data.to(self.device)
        self.value_net.to(self.device)

    def train(self):
        self.value_net.train()

    def eval(self):
        self.value_net.eval()

    def reset(self, idx=None):
        _ = idx

    def state_dict(self):
        return {
            "theta": self.theta.detach().cpu(),
            "value_net": self.value_net.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        _ = strict
        self.theta.data = state_dict["theta"].to(self.device)
        self.value_net.load_state_dict(state_dict["value_net"])
        self.policy_opt.load_state_dict(state_dict["policy_opt"])
        self.value_opt.load_state_dict(state_dict["value_opt"])

    @property
    def beta(self) -> float:
        return self.policy_beta

    def _dist_from_obj_vals(self, obj_vals: np.ndarray) -> Categorical:
        obj_t = torch.as_tensor(obj_vals, dtype=torch.float32, device=self.device)
        logits = _stable_categorical_logits(
            obj_t,
            beta=self.policy_beta,
            normalize_obj_values=self.normalize_obj_values,
            obj_norm_eps=self.obj_norm_eps,
        )
        return Categorical(logits=logits)

    def _dist_from_linearized_obj(self, step: PPOStep) -> Categorical:
        obj_t = torch.as_tensor(step.obj_vals, dtype=torch.float32, device=self.device)
        grad_t = torch.as_tensor(
            step.theta_grads, dtype=torch.float32, device=self.device
        )
        theta_ref_t = torch.as_tensor(
            step.theta_ref, dtype=torch.float32, device=self.device
        )
        theta_delta = self.theta - theta_ref_t
        obj_lin = obj_t + grad_t @ theta_delta
        logits = _stable_categorical_logits(
            obj_lin,
            beta=self.policy_beta,
            normalize_obj_values=self.normalize_obj_values,
            obj_norm_eps=self.obj_norm_eps,
        )
        return Categorical(logits=logits)

    def act(self, state: np.ndarray, deterministic: bool = False):
        """Select action from the existing MILP problem node via solver pool."""
        self._sync_model_params_from_theta()
        self.model.update_state(state)
        node = self.model.get_LP_formulation()
        sol_pool = self.solver.solve(node)
        if not sol_pool:
            return None, None

        obj_vals = np.asarray([sol["fun"] for sol in sol_pool], dtype=np.float32)
        dist = self._dist_from_obj_vals(obj_vals)

        if deterministic:
            action_idx = int(torch.argmax(dist.logits).item())
        else:
            action_idx = int(dist.sample().item())

        chosen = sol_pool[action_idx]
        action = chosen["x"][self.model.get_desc_var_indices()]
        if chosen.get("fathomed"):
            if self.nn_sample:
                action, _ = knn_branch_sample(action, chosen["bounds"])
            else:
                action, _ = naive_branch_sample(action, chosen["bounds"])

        actions = [
            np.asarray(sol["x"][self.model.get_desc_var_indices()], dtype=np.float32)
            for sol in sol_pool
        ]
        ineq_margs = [
            np.asarray(sol.get("ineqlin", []), dtype=np.float32) for sol in sol_pool
        ]
        eq_margs = [
            np.asarray(sol.get("eqlin", []), dtype=np.float32) for sol in sol_pool
        ]
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        theta_grads = np.asarray(
            [
                self.model.lagrange_gradient(a, state_arr, eq_marg, ineq_marg)
                for a, ineq_marg, eq_marg in zip(actions, ineq_margs, eq_margs)
            ],
            dtype=np.float32,
        )
        # Model gradients may include extra components (e.g. c); PPO theta uses only (aA, aB, b).
        if theta_grads.shape[1] != self.theta_dim:
            theta_grads = theta_grads[:, -self.theta_dim :]

        state_t = _as_state_tensor(state, self.device)
        value = float(self.value_net(state_t.unsqueeze(0)).squeeze(0).item())
        logp = float(dist.log_prob(torch.tensor(action_idx, device=self.device)).item())

        info = {
            "action_idx": action_idx,
            "obj_vals": obj_vals,
            "actions": np.asarray(actions, dtype=np.float32),
            "theta_grads": theta_grads,
            "theta_ref": self.theta.detach().cpu().numpy().astype(np.float32).copy(),
            "old_logp": logp,
            "value": value,
            "n_sols": len(sol_pool),
            "beta": float(self.beta),
        }
        return action, info

    def _compute_policy_terms(self, steps, advantages):
        ratio_terms = []
        unclipped_terms = []
        entropy_terms = []
        kls = []
        clip_active = []
        for i, step in enumerate(steps):
            dist = self._dist_from_linearized_obj(step)
            a_idx = torch.tensor(step.action_idx, dtype=torch.long, device=self.device)
            new_logp = dist.log_prob(a_idx)
            old_logp = torch.tensor(
                step.old_logp, dtype=torch.float32, device=self.device
            )

            ratio = torch.exp(new_logp - old_logp)
            adv_i = advantages[i]
            surr1 = ratio * adv_i
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_i
            )
            unclipped_terms.append(surr1)
            ratio_terms.append(torch.min(surr1, surr2))
            entropy_terms.append(dist.entropy())
            kls.append(old_logp - new_logp)
            clip_active.append(
                (
                    (ratio < (1.0 - self.clip_param))
                    | (ratio > (1.0 + self.clip_param))
                ).float()
            )

        clipped_obj = torch.stack(ratio_terms).mean()
        unclipped_obj = torch.stack(unclipped_terms).mean()
        policy_loss = -clipped_obj
        entropy = torch.stack(entropy_terms).mean()
        approx_kl = torch.stack(kls).mean()
        clip_fraction = torch.stack(clip_active).mean()
        clip_gap = unclipped_obj - clipped_obj
        return (
            policy_loss,
            entropy,
            approx_kl,
            unclipped_obj,
            clipped_obj,
            clip_gap,
            clip_fraction,
        )

    def update(
        self, buffer: PPORolloutBuffer, last_value: float = 0.0
    ) -> dict[str, float]:
        if len(buffer) == 0:
            raise ValueError("PPO buffer is empty")

        theta_before_update = self.theta.detach().clone()

        rewards_np = np.asarray([s.reward for s in buffer.steps], dtype=np.float32)
        dones = [s.done for s in buffer.steps]
        values = [s.value for s in buffer.steps]
        values_np = np.asarray(values, dtype=np.float32)

        rewards_train_np = rewards_np.copy()
        if self.minimize_env_reward:
            rewards_train_np = -rewards_train_np
        if self.normalize_rewards and rewards_train_np.size > 1:
            rewards_train_np = (rewards_train_np - rewards_train_np.mean()) / (
                rewards_train_np.std() + self.reward_norm_eps
            )
        if self.reward_clip is not None:
            rewards_train_np = np.clip(
                rewards_train_np, -self.reward_clip, self.reward_clip
            )

        returns_np, adv_np = compute_returns_and_advantages(
            rewards=rewards_train_np.tolist(),
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=float(last_value),
        )

        results = defaultdict(list)
        n_steps = len(buffer)
        mb_size = min(self.mini_batch_size, n_steps)
        step_returns = {i: float(returns_np[i]) for i in range(n_steps)}
        step_advs = {i: float(adv_np[i]) for i in range(n_steps)}

        # Per-update diagnostics from rollout statistics.
        adv_mean = float(np.mean(adv_np)) if adv_np.size > 0 else 0.0
        adv_std = float(np.std(adv_np)) if adv_np.size > 0 else 0.0
        adv_snr = float(abs(adv_mean) / (adv_std + 1e-8)) if adv_np.size > 0 else 0.0
        adv_abs_mean = float(np.mean(np.abs(adv_np))) if adv_np.size > 0 else 0.0
        adv_p10 = float(np.percentile(adv_np, 10.0)) if adv_np.size > 0 else 0.0
        adv_p50 = float(np.percentile(adv_np, 50.0)) if adv_np.size > 0 else 0.0
        adv_p90 = float(np.percentile(adv_np, 90.0)) if adv_np.size > 0 else 0.0
        adv_pos_ratio = float(np.mean(adv_np > 0.0)) if adv_np.size > 0 else 0.0
        returns_var = float(np.var(returns_np)) if returns_np.size > 0 else 0.0
        if returns_var > 1e-12:
            explained_var = 1.0 - float(np.var(returns_np - values_np) / returns_var)
        else:
            explained_var = 0.0

        if (
            adv_np.size > 1
            and returns_np.size > 1
            and np.std(adv_np) > 1e-12
            and np.std(returns_np) > 1e-12
        ):
            adv_returns_corr = float(np.corrcoef(adv_np, returns_np)[0, 1])
        else:
            adv_returns_corr = 0.0

        td_errors = adv_np.copy()
        td_error_mean = float(np.mean(td_errors)) if td_errors.size > 0 else 0.0
        td_error_std = float(np.std(td_errors)) if td_errors.size > 0 else 0.0

        mismatch_flags = []
        mismatch_l2 = []
        for s in buffer.steps:
            if s.chosen_action_raw is None or s.executed_action is None:
                continue
            raw = np.asarray(s.chosen_action_raw, dtype=np.float32).reshape(-1)
            exe = np.asarray(s.executed_action, dtype=np.float32).reshape(-1)
            mismatch_flags.append(float(not np.allclose(raw, exe, atol=1e-6)))
            mismatch_l2.append(float(np.linalg.norm(raw - exe)))
        action_mismatch_rate = (
            float(np.mean(mismatch_flags)) if len(mismatch_flags) > 0 else 0.0
        )
        action_mismatch_l2_mean = (
            float(np.mean(mismatch_l2)) if len(mismatch_l2) > 0 else 0.0
        )

        for _ in range(self.update_epochs):
            p_epoch = 0.0
            v_epoch = 0.0
            e_epoch = 0.0
            kl_epoch = 0.0
            unclipped_epoch = 0.0
            clipped_epoch = 0.0
            clip_gap_epoch = 0.0
            clip_frac_epoch = 0.0
            n_mb = 0

            for mb_idx, mb in buffer.sampler(mb_size, drop_last=False):
                adv_t = torch.as_tensor(
                    [step_advs[i] for i in mb_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.normalize_adv and adv_t.numel() > 1:
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                ret_t = torch.as_tensor(
                    [step_returns[i] for i in mb_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                states_t = torch.stack(
                    [_as_state_tensor(s.state, self.device) for s in mb], dim=0
                )

                (
                    policy_loss,
                    entropy,
                    approx_kl,
                    unclipped_obj,
                    clipped_obj,
                    clip_gap,
                    clip_fraction,
                ) = self._compute_policy_terms(mb, adv_t)
                value_pred = self.value_net(states_t)
                value_loss = 0.5 * (value_pred - ret_t).pow(2).mean()

                if (
                    self.target_kl is not None
                    and self.target_kl > 0
                    and approx_kl.detach().item() > 1.5 * self.target_kl
                ):
                    continue

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                total_loss.backward()
                self.policy_opt.step()
                self.value_opt.step()

                p_epoch += float(policy_loss.detach().cpu().item())
                v_epoch += float(value_loss.detach().cpu().item())
                e_epoch += float(entropy.detach().cpu().item())
                kl_epoch += float(approx_kl.detach().cpu().item())
                unclipped_epoch += float(unclipped_obj.detach().cpu().item())
                clipped_epoch += float(clipped_obj.detach().cpu().item())
                clip_gap_epoch += float(clip_gap.detach().cpu().item())
                clip_frac_epoch += float(clip_fraction.detach().cpu().item())
                n_mb += 1

            if n_mb == 0:
                continue

            results["policy_loss"].append(p_epoch / n_mb)
            results["value_loss"].append(v_epoch / n_mb)
            results["entropy_loss"].append(e_epoch / n_mb)
            results["approx_kl"].append(kl_epoch / n_mb)
            results["surrogate_unclipped"].append(unclipped_epoch / n_mb)
            results["surrogate_clipped"].append(clipped_epoch / n_mb)
            results["surrogate_clip_gap"].append(clip_gap_epoch / n_mb)
            results["clip_fraction"].append(clip_frac_epoch / n_mb)

        if not results:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "approx_kl": 0.0,
                "beta": float(self.beta),
                "theta_norm": float(
                    torch.linalg.norm(self.theta.detach()).cpu().item()
                ),
                "buffer_size": len(buffer),
                "minimize_env_reward": float(self.minimize_env_reward),
                "reward_raw_mean": float(rewards_np.mean()),
                "reward_raw_std": float(rewards_np.std()),
                "reward_train_mean": float(rewards_train_np.mean()),
                "reward_train_std": float(rewards_train_np.std()),
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "adv_snr": adv_snr,
                "adv_abs_mean": adv_abs_mean,
                "adv_p10": adv_p10,
                "adv_p50": adv_p50,
                "adv_p90": adv_p90,
                "adv_pos_ratio": adv_pos_ratio,
                "value_explained_variance": explained_var,
                "adv_returns_corr": adv_returns_corr,
                "td_error_mean": td_error_mean,
                "td_error_std": td_error_std,
                "action_mismatch_rate": action_mismatch_rate,
                "action_mismatch_l2_mean": action_mismatch_l2_mean,
                "surrogate_unclipped": 0.0,
                "surrogate_clipped": 0.0,
                "surrogate_clip_gap": 0.0,
                "clip_fraction": 0.0,
                "theta_update_norm": float(
                    torch.linalg.norm(self.theta.detach() - theta_before_update)
                    .cpu()
                    .item()
                ),
            }

        out = {k: float(sum(v) / len(v)) for k, v in results.items()}
        out["beta"] = float(self.beta)
        out["theta_norm"] = float(torch.linalg.norm(self.theta.detach()).cpu().item())
        out["buffer_size"] = len(buffer)
        out["minimize_env_reward"] = float(self.minimize_env_reward)
        out["reward_raw_mean"] = float(rewards_np.mean())
        out["reward_raw_std"] = float(rewards_np.std())
        out["reward_train_mean"] = float(rewards_train_np.mean())
        out["reward_train_std"] = float(rewards_train_np.std())
        out["adv_mean"] = adv_mean
        out["adv_std"] = adv_std
        out["adv_snr"] = adv_snr
        out["adv_abs_mean"] = adv_abs_mean
        out["adv_p10"] = adv_p10
        out["adv_p50"] = adv_p50
        out["adv_p90"] = adv_p90
        out["adv_pos_ratio"] = adv_pos_ratio
        out["value_explained_variance"] = explained_var
        out["adv_returns_corr"] = adv_returns_corr
        out["td_error_mean"] = td_error_mean
        out["td_error_std"] = td_error_std
        out["action_mismatch_rate"] = action_mismatch_rate
        out["action_mismatch_l2_mean"] = action_mismatch_l2_mean
        out["theta_update_norm"] = float(
            torch.linalg.norm(self.theta.detach() - theta_before_update).cpu().item()
        )
        self._sync_model_params_from_theta()
        return out


class PPO_MILP_Agent(PPOMILPAgent):
    """Compatibility class matching naming style from PPO-MPC utilities."""
