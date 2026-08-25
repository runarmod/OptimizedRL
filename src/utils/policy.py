from copy import copy, deepcopy
from itertools import product

import numpy as np
import torch


def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def isint(x):
    return int(x) == x


# def naive_branch_sample(sol,conds,action_size,bounds): # This doesnt handle if there are several conditions on a variable
#     """
#     """
#     action = np.zeros(shape = (action_size,))
#     vars = []
#     for var,_,val in conds:
#         vars.append(var)
#         action[var] = sol[var] if isint(sol[var]) else val

#     for i in range(action_size):
#         bound = bounds[i]
#         action[i] = np.random.randint(bound[0],bound[1]) if i not in vars else action[i]
#     return action


def naive_branch_sample(sol, bounds):
    action = np.zeros_like(sol)
    new_bounds = copy(bounds)
    for i, var in enumerate(sol):
        if isint(var):
            action[i] = var
        else:
            action[i] = np.random.randint(bounds[i][0], bounds[i][1] + 1)
        new_bounds[i] = (
            action[i],
            action[i],
        )  # dont know if this should be fixed for all vars
    return action, new_bounds


def nn_branch_sample(sol, bounds):
    action = np.zeros_like(sol)
    new_bounds = copy(bounds)
    for i, var in enumerate(sol):
        if isint(var):
            action[i] = var
        else:
            action[i] = max(min(round(var), bounds[i][1]), bounds[i][0])
        new_bounds[i] = (action[i], action[i])
    return action, new_bounds


def naive_branch_sample_only_keep_ints(
    sol, conds, action_size, bounds
):  # This doesnt handle if there are several conditions on a variable
    """
    Not fixed at condition
    """
    action = np.zeros(shape=(action_size,))

    # for i in range(len(sol)):
    #     if isint(sol[i]):
    #         action[i] = sol[i]
    #     elif
    vars = []
    for var, comp, val in conds:
        vars.append(var)
        if isint(sol[var]):
            action[var] = sol[var]
        elif comp == "<=":
            action[var] = np.random.randint(bounds[var][0], val)
        elif comp == ">=":
            action[var] = np.random.randint(val, bounds[var][1])

    for i in range(action_size):
        bound = bounds[i]
        action[i] = (
            np.random.randint(bound[0], bound[1]) if i not in vars else action[i]
        )
    return action


def nn_branch_sample_only_keep_ints(
    sol, conds, action_size, bounds
):  # This doesnt handle if there are several conditions on a variable
    """
    Not fixed at condition
    """
    action = np.zeros(shape=(action_size,))
    vars = []
    for var, comp, val in conds:
        vars.append(var)
        if isint(sol[var]):
            action[var] = sol[var]
        elif comp == "<=":
            action[var] = max(min(round(sol[var]), val), bounds[var][0])
        elif comp == ">=":
            action[var] = max(min(round(sol[var]), bounds[var][1]), val)

    for i in range(action_size):
        bound = bounds[i]
        action[i] = (
            max(min(round(sol[i]), bound[1]), bound[0]) if i not in vars else action[i]
        )
    return action


# Addition to the original thesis codebase


def knn_branch_sample_simple(sol, bounds, k=3):
    action = np.zeros_like(sol, dtype=int)
    new_bounds = copy(bounds)

    candidates_per_var = []
    for i, var in enumerate(sol):
        if isint(var):
            action[i] = int(var)
            candidates_per_var.append([int(var)])
        else:
            center = round(var)
            half = k // 2
            low = max(int(bounds[i][0]), center - half)
            high = min(int(bounds[i][1]), center + half)
            candidates = list(range(low, high + 1))
            if len(candidates) < k:
                left = max(int(bounds[i][0]), center - k + 1)
                right = min(int(bounds[i][1]), center + k - 1)
                candidates = list(range(left, right + 1))
                candidates = [
                    c
                    for c in candidates
                    if c >= int(bounds[i][0]) and c <= int(bounds[i][1])
                ]
            candidates_per_var.append(candidates)
            action[i] = int(np.random.choice(candidates))
        new_bounds[i] = (action[i], action[i])
    return action, new_bounds


def knn_branch_sample(sol, bounds, k=9, max_pts=5000):

    dims = len(np.array(sol))

    temp_bounds = []
    for i in range(dims):
        if i < len(bounds):
            low, high = bounds[i]
        else:
            low, high = None, None

        if low is None:
            low = np.floor(sol[i] - k)
        if high is None:
            high = np.ceil(sol[i] + k)

        temp_bounds.append((low, high))

    ranges = [range(int(low), int(high) + 1) for (low, high) in temp_bounds]
    total_space = np.prod([len(r) for r in ranges])

    if total_space > max_pts:
        ranges = [
            range(
                max(int(low), int(np.floor(s - k))),
                min(int(high), int(np.ceil(s + k))) + 1,
            )
            for s, (low, high) in zip(sol, temp_bounds)
        ]

    all_points = np.array(list(product(*ranges)))
    dist = np.sum(np.abs(all_points - sol), axis=1)

    idx = np.argsort(dist)[:k]
    nearest_points = all_points[idx]
    nearest_dist = dist[idx]

    beta = 0.5
    weights = np.exp(-beta * nearest_dist)
    weights /= np.sum(weights)

    idx_choice = np.random.choice(len(nearest_points), p=weights)
    action = nearest_points[
        idx_choice
    ]  # dette blir 1D-array med samme dimensjon som sol

    new_bounds = deepcopy(bounds)

    for i, val in enumerate(action):
        new_bounds[i] = (val, val)

    return action, new_bounds


## End addition

# def nn_branch_sample(sol,conds,action_size,bounds): # This doesnt handle if there are several conditions on a variable
#     action = np.zeros(shape = (action_size,))
#     vars = []
#     for var,_,val in conds:
#         vars.append(var)
#         action[var] = sol[var] if isint(sol[var]) else val

#     for i in range(action_size):
#         bound = bounds[i]
#         action[i] = max(min(round(sol[i]),bound[1]),bound[0]) if i not in vars  else action[i]
#     return action


def policy_dist(obj_vals, beta=1):
    exps = np.exp((-1) * beta * obj_vals)
    alpha = np.sum(exps)
    return exps / alpha


def policy_dist_np(obj_vals, beta=1):

    return np.exp((-1) * beta * obj_vals - logsumnp((-1) * beta * obj_vals))


# def policy_dist_torch(obj_vals,beta = 1):
#     exps = torch.exp((-1)*beta*obj_vals)
#     alpha = torch.sum(exps)
#     return torch.divide(exps,alpha)


def policy_dist_torch(obj_vals, beta=1):

    return torch.exp((-1) * beta * obj_vals - logsumtorch((-1) * beta * obj_vals))


def logsumtorch(x):
    c = torch.max(x)
    return c + torch.log(torch.sum(torch.exp(x - c)))


def logsumnp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


# This math could be off
def nabla_log_pi(action_taken_object_grad, obj_vals, obj_grads, beta=1):
    # print("obj:",obj_vals)
    # print("grads:",np.array(obj_grads))
    exps = np.exp(-beta * obj_vals)
    alpha = np.sum(exps)
    # upper_right = np.sum(beta*exps*obj_grads)
    upper_right = beta * (exps @ obj_grads)
    right_side = upper_right / alpha
    left_side = beta * action_taken_object_grad
    return right_side - left_side


# Might just make one bigger func


def policy_grad(nabla_log_pi, adv):
    return np.sum(nabla_log_pi * adv)


def nabla_log_pi_stable(action_taken_object_grad, obj_vals, obj_grads, beta=1):
    """
    Computes the gradient of the log-softmax probability for the taken action
    with temperature beta, using a numerically stable method.

    Args:
        action_taken_object_grad: The gradient of the objective function value
                                   for the action that was actually taken.
        obj_vals (np.ndarray): Array of objective function values for all possible actions.
        obj_grads (list or np.ndarray): List or array of gradients of the objective
                                        function values for all possible actions.
                                        If it's a list of arrays, it will be converted.
                                        Assumes obj_grads[i] corresponds to obj_vals[i].
        beta (float): The inverse temperature parameter.

    Returns:
        np.ndarray: The gradient of the log-probability of the taken action.
    """
    # Ensure obj_vals is a numpy array for vectorized operations
    obj_vals = np.asarray(obj_vals, dtype=float)

    # Ensure obj_grads is a numpy array for vectorized operations
    # Assuming obj_grads is a list of vectors (or a 2D array where rows are gradients)
    obj_grads_arr = np.array(obj_grads)
    if obj_grads_arr.ndim == 1 and len(obj_vals) > 1:
        # Handle case where gradients might be scalars but there are multiple actions
        # This might indicate an issue or a specific use case.
        # Assuming here obj_grads should align with obj_vals shape in the first dim.
        # If gradients are truly scalar, reshape might be needed depending on desired output shape.
        # For typical policy gradients, grads are vectors of same dim as parameters.
        pass  # No change needed if grads are correctly shaped (N_actions, N_params) or similar
    elif obj_grads_arr.ndim == 0 and len(obj_vals) == 1:
        # Handle single action case
        obj_grads_arr = (
            obj_grads_arr.reshape(1, -1)
            if obj_grads_arr.ndim > 0
            else np.array([obj_grads_arr])
        )

    if obj_vals.size == 0:
        # Handle edge case of empty inputs if necessary
        # This might return 0, raise error, or depend on context.
        # Assuming action_taken_object_grad would also be appropriately sized or zero.
        return np.zeros_like(action_taken_object_grad)

    # 1. Calculate x_j = -beta * O_j
    neg_beta_obj_vals = -beta * obj_vals

    # 2. Find the maximum c = max(x_j)
    #    Subtracting the max prevents overflow and helps with underflow.
    max_val = np.max(neg_beta_obj_vals)

    # 3. Calculate stabilized exponentials E_j = exp(x_j - c)
    stable_exps = np.exp(neg_beta_obj_vals - max_val)

    # 4. Calculate the sum S = sum(E_l)
    sum_stable_exps = np.sum(stable_exps)

    # Check for potential division by zero if all exps underflowed despite the trick
    # (This is less likely now but good practice)
    if sum_stable_exps == 0:
        # Handle this case: perhaps return zero gradient or a default behavior
        # This could happen if beta is very large and differences in obj_vals are huge.
        # Or if all obj_vals are -inf.
        # A simple approach might be uniform probability, but the gradient is complex.
        # Returning zero might be a safe fallback if this state is unexpected.
        # The gradient contribution from the expectation term would be ill-defined.
        # We might approximate as only the action_taken gradient matters.
        # return -beta * action_taken_object_grad # This corresponds to pi_k=1
        # Or assume uniform prob:
        num_actions = len(obj_vals)
        avg_grad = np.mean(obj_grads_arr, axis=0)
        return beta * (avg_grad - action_taken_object_grad)

    # 5. Calculate stable softmax probabilities pi_j = E_j / S
    softmax_probs = stable_exps / sum_stable_exps

    # 6. Compute the weighted sum of gradients: G_avg = sum(pi_j * g_j)
    #    Use np.dot or matmul (@) if obj_grads_arr is (N_actions, N_params)
    #    Use np.tensordot or einsum for more complex tensor shapes if needed.
    #    Assuming obj_grads_arr is (N_actions, N_params):
    if obj_grads_arr.ndim > 1:
        # weighted_grads_sum = np.dot(softmax_probs, obj_grads_arr) # Alternative
        weighted_grads_sum = softmax_probs @ obj_grads_arr
    else:  # Handle case where gradients are scalars
        weighted_grads_sum = np.dot(softmax_probs, obj_grads_arr)

    # 7. Compute the final gradient: beta * (G_avg - g_k)
    #    Ensure action_taken_object_grad is a numpy array for subtraction.
    action_taken_grad_arr = np.asarray(action_taken_object_grad)

    gradient = beta * (weighted_grads_sum - action_taken_grad_arr)

    return gradient


# obj_vals = np.random.rand(1,10)
# # obj_vals = [i for i in range(10)]
# obj_grads=  np.random.rand(1,10)

# nab = nabla_log_pi(obj_vals,obj_grads)

# adv = np.random.rand(1,10)

# print(policy_grad(nab,adv))


if __name__ == "__main__":
    pass
