import numpy as np
import torch
from scipy.optimize import linprog
from tqdm import tqdm

from src.utils.policy import (
    categorical,
    nabla_log_pi_stable,
    naive_branch_sample,
    nn_branch_sample,
    policy_dist_np,
)


def dLdx(c, A_ub, A_eq, ineq, eq, upper, lower):
    A_ub = np.array([]) if A_ub is None else A_ub
    A_eq = np.array([]) if A_eq is None else A_eq
    return c - ineq @ A_ub - eq @ A_eq - upper - lower


def calc_actual_grad(node):
    sol = linprog(
        node["c"],
        node["A_ub"],
        node["b_ub"],
        node["A_eq"],
        node["b_eq"],
        node["bounds"],
    )
    ineq = sol.ineqlin.marginals
    eq = sol.ineqlin.marginals
    return ineq, eq


class PPOActor:
    def __init__(
        self,
        model,
        solver,
        critic,
        beta=1,
        lr=0.01,
        df=0.9,
        nn_sample=False,
        sampled_grad=False,
        clip_param=0.2,
        target_kl=0.02,
        entropy_coef=0.0,
        ppo_epochs=10,
        mini_batch_size=64,
        min_grad_diversity=1e-8,
    ):
        self.model = model
        self.desc_vars = self.model.get_desc_var_indices()
        self.n_desc_vars = self.model.n_desc_vars
        self.buffer = ExperienceBufferPPO()
        self.lr = lr
        self.df = df
        self.beta = beta
        self.solver = solver
        self.critic = critic
        self.nn_sample = nn_sample
        self.sampled_grad = sampled_grad

        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.min_grad_diversity = min_grad_diversity

        self.last_approx_kl = 0.0
        self.last_entropy = 0.0

    def act(self, new_state):
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation()

        sol_pool = self.solver.solve(node)
        if sol_pool is None:
            return None

        for sol in sol_pool:
            if np.any(
                np.abs(
                    dLdx(
                        node["c"],
                        node["A_ub"],
                        node["A_eq"],
                        sol["ineqlin"],
                        sol["eqlin"],
                        sol["upper"],
                        sol["lower"],
                    )
                )
                > 1e-4
            ):
                raise Exception("dLdx isnt 0")

        obj_values = np.array([sol["fun"] for sol in sol_pool])
        pol = policy_dist_np(obj_values, self.beta)

        draw = categorical(pol)
        chosen_sol = sol_pool[draw]
        bounds = chosen_sol["bounds"]

        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action, bounds = nn_branch_sample(
                    chosen_sol["x"][self.desc_vars], bounds
                )
            else:
                action, bounds = naive_branch_sample(
                    chosen_sol["x"][self.desc_vars], bounds
                )
        else:
            action = chosen_sol["x"][self.desc_vars]

        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [np.array(sol["ineqlin"]) for sol in sol_pool]
        eq_margs = [np.array(sol["eqlin"]) for sol in sol_pool]
        lag_grads = [
            self.model.lagrange_gradient(a, new_state, eq_marg, ineq_marg)
            for a, ineq_marg, eq_marg in zip(actions, ineq_margs, eq_margs)
        ]

        if self.sampled_grad:
            node["bounds"] = bounds
            ineq, eq = calc_actual_grad(node)
            lag_grads[draw] = self.model.lagrange_gradient(action, new_state, ineq, eq)

        lag_grads = np.array(lag_grads)
        lag_grad_action_drawn = lag_grads[draw]
        nab = nabla_log_pi_stable(
            lag_grad_action_drawn, obj_values, lag_grads, self.beta
        )

        grad_diversity = float(np.mean(np.var(lag_grads, axis=0)))

        info = {
            "fathomed": chosen_sol["fathomed"],
            "nab": nab,
            "n_sols": len(sol_pool),
            "t_nab": 0,
            "logp_old": float(np.log(np.clip(pol[draw], 1e-12, 1.0))),
            "draw": int(draw),
            "chosen_desc_action": np.array(chosen_sol["x"][self.desc_vars], copy=True),
            "grad_diversity": grad_diversity,
            "entropy": float(-np.sum(pol * np.log(np.clip(pol, 1e-12, 1.0)))),
        }
        return action, info

    def _match_solution_index(self, sol_pool, chosen_desc_action):
        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        if len(actions) == 0:
            return None
        actions_arr = np.asarray(actions)
        dists = np.linalg.norm(actions_arr - chosen_desc_action, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > 1e-8:
            return None
        return idx

    def _evaluate_current_policy_sample(self, state, chosen_desc_action):
        self.model.update_state(state)
        node = self.model.get_LP_formulation()
        sol_pool = self.solver.solve(node)
        if sol_pool is None or len(sol_pool) == 0:
            return None

        draw = self._match_solution_index(sol_pool, chosen_desc_action)
        if draw is None:
            return None

        obj_values = np.array([sol["fun"] for sol in sol_pool])
        pol = policy_dist_np(obj_values, self.beta)
        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [np.array(sol["ineqlin"]) for sol in sol_pool]
        eq_margs = [np.array(sol["eqlin"]) for sol in sol_pool]
        lag_grads = [
            self.model.lagrange_gradient(a, state, eq_marg, ineq_marg)
            for a, ineq_marg, eq_marg in zip(actions, ineq_margs, eq_margs)
        ]
        lag_grads = np.asarray(lag_grads)
        lag_grad_action_drawn = lag_grads[draw]
        nab = nabla_log_pi_stable(
            lag_grad_action_drawn, obj_values, lag_grads, self.beta
        )

        return {
            "nab": nab,
            "logp": float(np.log(np.clip(pol[draw], 1e-12, 1.0))),
            "entropy": float(-np.sum(pol * np.log(np.clip(pol, 1e-12, 1.0)))),
            "grad_diversity": float(np.mean(np.var(lag_grads, axis=0))),
        }

    def train(self, iters=1, sample=False, num_samples=0.5):
        size = len(self.buffer.rewards)
        if size == 0:
            raise Exception("Buffers are empty")

        pol_grad = None
        for _ in tqdm(range(max(1, iters)), leave=False, desc="Training"):
            if sample:
                indexes = np.array(range(int(size * num_samples)))
                np.random.shuffle(indexes)
            else:
                indexes = np.array(range(size))
            indexes = indexes.astype(int)

            rewards = np.array(self.buffer.rewards)[indexes]
            actions = np.array(self.buffer.actions)[indexes]
            states = np.array(self.buffer.states, dtype=float)[indexes]
            nxt_states = torch.tensor(np.array(self.buffer.nxt_states, dtype=float))[
                indexes
            ]

            self.critic.train(rewards, actions, states, nxt_states)
            advantages = self.critic.evaluate(actions, states, rewards, nxt_states)
            advantages = np.asarray(advantages, dtype=float).reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_logps = np.asarray(self.buffer.logp_old)[indexes]
            chosen_desc_actions = np.asarray(
                self.buffer.chosen_desc_actions, dtype=float
            )[indexes]

            n_data = len(indexes)
            if n_data == 0:
                continue

            approx_kl_epoch = 0.0
            entropy_epoch = 0.0
            used_count_epoch = 0

            ppo_epochs = max(1, self.ppo_epochs)
            mini_batch_size = max(1, min(self.mini_batch_size, n_data))
            for _epoch in range(ppo_epochs):
                shuffled = np.random.permutation(n_data)
                early_stop = False
                for start in range(0, n_data, mini_batch_size):
                    mb = shuffled[start : start + mini_batch_size]
                    batch_grad = None
                    used_count = 0
                    approx_kl_mb = 0.0
                    entropy_mb = 0.0

                    for bi in mb:
                        sample_eval = self._evaluate_current_policy_sample(
                            states[bi], chosen_desc_actions[bi]
                        )
                        if sample_eval is None:
                            continue
                        if sample_eval["grad_diversity"] < self.min_grad_diversity:
                            continue

                        logp_old = old_logps[bi]
                        logp_new = sample_eval["logp"]
                        ratio = float(np.exp(np.clip(logp_new - logp_old, -20.0, 20.0)))
                        adv = float(advantages[bi])

                        clipped_ratio = float(
                            np.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        )
                        unclipped = ratio * adv
                        clipped = clipped_ratio * adv
                        active = (
                            (unclipped <= clipped)
                            if adv >= 0
                            else (unclipped >= clipped)
                        )

                        scale = -ratio * adv if active else 0.0
                        grad_i = scale * sample_eval["nab"]
                        if self.entropy_coef != 0.0:
                            grad_i += -self.entropy_coef * sample_eval["nab"]

                        if batch_grad is None:
                            batch_grad = np.zeros_like(grad_i)
                        batch_grad += grad_i
                        approx_kl_mb += logp_old - logp_new
                        entropy_mb += sample_eval["entropy"]
                        used_count += 1

                    if used_count == 0:
                        continue

                    batch_grad /= used_count
                    self.model.update_params(batch_grad, self.lr)
                    pol_grad = batch_grad

                    approx_kl_batch = approx_kl_mb / used_count
                    approx_kl_epoch += approx_kl_batch
                    entropy_epoch += entropy_mb / used_count
                    used_count_epoch += 1

                    if self.target_kl > 0 and approx_kl_batch > 1.5 * self.target_kl:
                        early_stop = True
                        break
                if early_stop:
                    break

            if used_count_epoch > 0:
                self.last_approx_kl = approx_kl_epoch / used_count_epoch
                self.last_entropy = entropy_epoch / used_count_epoch
            else:
                self.last_approx_kl = 0.0
                self.last_entropy = 0.0

            if pol_grad is None:
                pol_grad = np.zeros_like(np.asarray(self.buffer.nabs[0]))

        self.buffer.reset()
        return pol_grad

    def update_buffers(
        self,
        reward,
        action,
        state,
        new_state,
        nab,
        t_nab,
        logp_old=None,
        draw=None,
        chosen_desc_action=None,
    ):
        self.buffer.rewards.append(reward)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        self.buffer.nxt_states.append(new_state)
        self.buffer.nabs.append(nab)
        self.buffer.t_nabs.append(t_nab)
        self.buffer.logp_old.append(logp_old if logp_old is not None else 0.0)
        self.buffer.draws.append(draw if draw is not None else -1)
        if chosen_desc_action is None:
            self.buffer.chosen_desc_actions.append(np.array([], dtype=float))
        else:
            self.buffer.chosen_desc_actions.append(
                np.array(chosen_desc_action, copy=True)
            )


class ExperienceBufferPPO:
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.nxt_states = []
        self.nabs = []
        self.t_nabs = []
        self.logp_old = []
        self.draws = []
        self.chosen_desc_actions = []

    def reset(self):
        del self.rewards[:]
        del self.actions[:]
        del self.states[:]
        del self.nxt_states[:]
        del self.nabs[:]
        del self.t_nabs[:]
        del self.logp_old[:]
        del self.draws[:]
        del self.chosen_desc_actions[:]
