import numpy as np
import torch
from scipy.optimize import linprog
from tqdm import tqdm

from src.utils.policy import (
    categorical,
    knn_branch_sample,
    nabla_log_pi_stable,
    naive_branch_sample,
    policy_dist_np,
    policy_dist_torch,
)


def dLdx(c, A_ub, A_eq, ineq, eq, upper, lower):
    A_ub = np.array([]) if A_ub is None else A_ub
    A_eq = np.array([]) if A_eq is None else A_eq
    # return c - ineq @ A_ub - eq @ A_eq - upper + lower

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


def check_corr_grad(
    obj_vals, nab, beta, lag_grads, draw
):  # This doesnt work with sampled nab probably because htat one gradient is wrong specfically
    obj_vals_torch = torch.tensor(obj_vals, requires_grad=True)
    pol = policy_dist_torch(obj_vals_torch, beta)
    log_pol = torch.log(pol)
    log_pol[draw].backward()  # dpi/dphi
    grad_log_pol = obj_vals_torch.grad
    expected = grad_log_pol @ torch.as_tensor(lag_grads, dtype=grad_log_pol.dtype)
    assert np.linalg.norm(nab - expected.detach().numpy()) < 1e-6


class Actor:
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
    ):
        self.model = model

        self.desc_vars = self.model.get_desc_var_indices()
        self.lag_grads = None
        self.lag_grad_action_drawn = None
        self.n_desc_vars = self.model.n_desc_vars
        self.nab = None
        self.buffer = ExperienceBuffer()
        self.lr = lr
        self.df = df
        self.beta = beta
        self.solver = solver
        self.critic = critic
        self.value_est = 0
        self.nn_sample = nn_sample
        self.sampled_grad = sampled_grad
        self.m = None
        self.v = None
        self.reset_critic_iter = 0

    def act(self, new_state):
        # Compute next action
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation()

        sol_pool = self.solver.solve(
            node
        )  # Has to return an array of dicts that include the action x, the obj func, and marginals

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
        # Sample unexplored nodes
        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action, bounds = knn_branch_sample(
                    chosen_sol["x"][self.desc_vars], bounds
                )
            else:
                action, bounds = naive_branch_sample(
                    chosen_sol["x"][self.desc_vars], bounds
                )

        else:
            action = chosen_sol["x"]
            action = action[self.desc_vars]

        # Compute model specific gradient

        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [
            np.array(sol["ineqlin"]) for sol in sol_pool
        ]  # negative signs here could be wrong
        eq_margs = [
            np.array(sol["eqlin"]) for sol in sol_pool
        ]  # negative signs here could be wrong
        lag_grads = [
            self.model.lagrange_gradient(a, new_state, eq_marg, ineq_marg)
            for a, ineq_marg, eq_marg in zip(actions, ineq_margs, eq_margs)
        ]

        # Convert action solution to actual action
        if self.sampled_grad:
            node["bounds"] = bounds
            ineq, eq = calc_actual_grad(node)
            lag_grad_action_drawn = self.model.lagrange_gradient(
                action, new_state, ineq, eq
            )
            lag_grads[draw] = lag_grad_action_drawn
        lag_grads = np.array(lag_grads)
        # Compute policy sensitivity
        lag_grad_action_drawn = lag_grads[draw]
        # old_nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        nab = nabla_log_pi_stable(
            lag_grad_action_drawn, obj_values, lag_grads, self.beta
        )
        check_corr_grad(obj_values, nab, self.beta, lag_grads, draw)

        t_nab = 0

        info = {
            "fathomed": chosen_sol["fathomed"],
            "nab": nab,
            "n_sols": len(sol_pool),
            "t_nab": t_nab,
        }
        return action, info

    def train(self, iters=1000, sample=False, num_samples=0.5):

        # Not sure how q_table should be trained

        size = len(self.buffer.rewards)

        if size == 0:
            raise Exception("Buffers are empty")
        for _ in tqdm(range(iters), leave=False, desc="Training"):
            if sample:
                indexes = np.array(range(int(size * num_samples)))
                np.random.shuffle(indexes)
            else:
                indexes = np.array(range(size))
            indexes = indexes.astype(int)
            rewards = np.array(self.buffer.rewards)[indexes]
            actions = np.array(self.buffer.actions)[indexes]
            states = np.array(self.buffer.states)[indexes]
            nxt_states = torch.tensor(self.buffer.nxt_states)[indexes]
            nabs = np.array(self.buffer.nabs)[indexes]
            t_nabs = np.array(self.buffer.t_nabs)[indexes]

            self.critic.train(rewards, actions, states, nxt_states)

            qualities = self.critic.evaluate(actions, states, rewards, nxt_states)

            pol_grad = ((nabs.T @ qualities) / len(rewards)).squeeze()
            if self.v is None:
                self.v = np.zeros_like(pol_grad)
            if self.m is None:
                self.m = np.zeros_like(pol_grad)

            self.model.update_params(pol_grad, self.lr)
        self.buffer.reset()
        return pol_grad

    def update_buffers(self, reward, action, state, new_state, nab, t_nab):
        self.buffer.rewards.append(reward)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        self.buffer.nxt_states.append(new_state)
        self.buffer.nabs.append(nab)
        self.buffer.t_nabs.append(t_nab)
        # self.buffer.nabs.append(self.nab)
        # # Make sure we dont use it twice :)
        # del self.nab


class ExperienceBuffer:
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.nxt_states = []
        self.nabs = []
        self.t_nabs = []

    def reset(self):
        del self.rewards[:]
        del self.actions[:]
        del self.states[:]
        del self.nxt_states[:]
        del self.nabs[:]
        del self.t_nabs[:]
