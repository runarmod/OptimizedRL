import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
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

# def categorical(p):
#     return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def dLdx(c,A_ub,A_eq,ineq,eq,upper,lower):
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
        node["bounds"]
    )
    ineq = sol.ineqlin.marginals
    eq = sol.ineqlin.marginals
    return ineq,eq
 


# def dLdx_2(c,aA,aB,b,ineq,upper,lower):
#     dLdxt = c -ineq @ aB - upper - lower 
#     # return c - ineq @ A_ub - eq @ A_eq - upper + lower

#     return c - ineq @ A_ub - eq @ A_eq - upper - lower


# vals = torch.rand((4,))
# vals_np = vals.numpy()
# vals.requires_grad = True
# dphidtheta = torch.rand((4,2)) #
# dphidtheta_np = dphidtheta.numpy()
# dphidtheta.requires_grad = True



# print(vals)
# pol = policy.policy_dist_torch(vals,1) # pi
# print(pol)
# log_pol = torch.log(pol)
# log_pol[2].backward() # dpi/dphi
# print("Gradient of policy w.r.t. objective values:",vals.grad)

# # jac = torch.autograd.functional.jacobian(policy.policy_dist_torch,vals)
# # print(jac)
# nab = policy.nabla_log_pi(dphidtheta_np[2],vals_np,dphidtheta_np,beta = 1) # dpi/dtheta
# print(nab)

# grad_log_pol = vals.grad  # This is d(log π) / dvals
# expected_nabla_log_pi = grad_log_pol @ dphidtheta
# print(expected_nabla_log_pi)
# error = np.linalg.norm(nab - expected_nabla_log_pi.detach().numpy())
# print(f"Error between manual and PyTorch gradients: {error}")
# print(nab,expected_nabla_log_pi)

def check_corr_grad(obj_vals,nab,beta,lag_grads,draw): # This doesnt work with sampled nab probably because htat one gradient is wrong specfically
    obj_vals_torch = torch.tensor(obj_vals,requires_grad= True)
    pol = policy_dist_torch(obj_vals_torch,beta)
    log_pol = torch.log(pol)
    log_pol[draw].backward() # dpi/dphi
    grad_log_pol = obj_vals_torch.grad
    expected = grad_log_pol @ torch.as_tensor(lag_grads, dtype=grad_log_pol.dtype)
    assert np.linalg.norm(nab - expected.detach().numpy()) < 1e-6


def check_with_cvxpylayers(node,bounds,lag_grad,drawn_x,state):
    c_b = torch.tensor(node["c"],requires_grad=True)  # Objective function

    A_ub_b = torch.tensor(node["A_ub"], requires_grad=True)
    b_ub_b = torch.tensor(node["b_ub"], requires_grad=True)

    solver_args = {
        'max_iters': 50000,  # Increase max iterations (default is often 2500)
        # 'eps': 1e-5,      # Adjust tolerance if needed (SCS default is 1e-4)
        # 'verbose': True,   # Set to True to get detailed solver output for debugging
        'solve_method' : 'ECOS'
    }
    lb_b = torch.tensor([bound[0] if bound[0] is not None else -1e3 for bound in bounds],dtype=torch.float64)
    ub_b = torch.tensor([bound[1] if bound[1] is not None else 1e3 for bound in bounds],dtype=torch.float64)
    x = cp.Variable(c_b.shape[0])
    c = cp.Parameter(c_b.shape[0])  # ✅ Change to Parameter
    A_ub = cp.Parameter(A_ub_b.shape)
    b_ub = cp.Parameter(b_ub_b.shape[0])
    lb = cp.Parameter(lb_b.shape)
    ub = cp.Parameter(ub_b.shape)

    constraints = [A_ub @ x <= b_ub, lb <= x, x <= ub]

    objective = cp.Minimize(c @ x)  # ✅ Uses c as a parameter
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  # ✅ Now this should pass

    cvxpylayer = CvxpyLayer(problem, parameters=[c,A_ub, b_ub,lb,ub], variables=[x])
    solution, = cvxpylayer(c_b, A_ub_b, b_ub_b,lb_b,ub_b,solver_args = solver_args)
    objective_value = (c_b @ solution)
    objective_value.backward()
    # assert all(torch.abs(c_b.grad[:3] - lag_grad[:3]) < 1e-3)
    if (
            not all(torch.abs(c_b.grad[:3] - lag_grad[:3]) < 1e-3)
            or  not all(torch.abs(A_ub_b.grad[:7,:3].flatten() - lag_grad[45:66]) < 1e-3)
            or not all(torch.abs(b_ub_b.grad[:7] + lag_grad[-7:]) < 1e-3) 
            or not all(torch.abs((b_ub_b.grad[:7].unsqueeze(0).T @ torch.tensor(state,dtype = torch.double).unsqueeze(0)).flatten() + lag_grad[3:45]) < 1e-3)
        ):
        pass
    true_grad = torch.hstack((c_b.grad[:3],A_ub_b.grad[:7,:3].flatten(),-b_ub_b.grad[:7],-(b_ub_b.grad[:7].unsqueeze(0).T @ torch.tensor(state,dtype = torch.double).unsqueeze(0)).flatten()))
    return true_grad
    


    # # Equality constraint: x1 + x2 = 2
    # A_eq_b = torch.tensor([[1, 1]], requires_grad=True, dtype=torch.float32)
    # b_eq_b = torch.tensor([2], requires_grad=True, dtype=torch.float32)





class Actor:
    def __init__(self,model,solver,critic,beta = 1,lr = 0.01,df = 0.9,nn_sample = False,sampled_grad = False):
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

    def act(self,new_state):
        # Compute next action
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation()

        sol_pool = self.solver.solve(node) # Has to return an array of dicts that include the action x, the obj func, and marginals

        if sol_pool is None:
            return None
        for sol in sol_pool:
            if np.any(np.abs(dLdx(node["c"],node["A_ub"],node["A_eq"],sol["ineqlin"],sol["eqlin"],sol["upper"],sol["lower"])) > 1e-4 ):
                raise Exception("dLdx isnt 0")

        obj_values =np.array( [sol["fun"] for sol in sol_pool])
        pol = policy_dist_np(obj_values,self.beta)

        draw = categorical(pol)
        chosen_sol = sol_pool[draw]
        bounds = chosen_sol["bounds"]
        # Sample unexplored nodes
        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action,bounds = knn_branch_sample(chosen_sol['x'][self.desc_vars],bounds)
            else:
                action,bounds = naive_branch_sample(chosen_sol['x'][self.desc_vars],bounds)

        else:
            action = chosen_sol["x"]
            action = action[self.desc_vars]





        # Compute model specific gradient

        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [np.array(sol["ineqlin"]) for sol in sol_pool] # negative signs here could be wrong
        eq_margs = [np.array(sol["eqlin"]) for sol in sol_pool] #negative signs here could be wrong
        lag_grads = [self.model.lagrange_gradient(a,new_state,eq_marg,ineq_marg) for a,ineq_marg,eq_marg in zip(actions,ineq_margs,eq_margs)]

        # Convert action solution to actual action
        if self.sampled_grad:
            node["bounds"] = bounds
            ineq,eq = calc_actual_grad(node)
            lag_grad_action_drawn = self.model.lagrange_gradient(action,new_state,ineq,eq)
            lag_grads[draw] = lag_grad_action_drawn            
        lag_grads = np.array(lag_grads)
        # Compute policy sensitivity
        lag_grad_action_drawn = lag_grads[draw]
        # old_nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        nab = nabla_log_pi_stable(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        check_corr_grad(obj_values,nab,self.beta,lag_grads,draw) 

        t_nab = 0

        info = {
            "fathomed" : chosen_sol["fathomed"],
            "nab" : nab,
            "n_sols" : len(sol_pool),
            "t_nab" : t_nab
        }
        return action,info

    def train(self,iters = 1000,sample = False,num_samples = 0.5):

        # Not sure how q_table should be trained

        size = len(self.buffer.rewards)

        if size == 0:
            raise Exception("Buffers are empty")
        for _ in tqdm(range(iters),leave=False,desc = "Training"):
            if sample:

                indexes = np.array(range(int(size*num_samples)))
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
         
            self.critic.train(rewards,actions,states,nxt_states)




            qualities = self.critic.evaluate(actions,states,rewards,nxt_states)

            pol_grad = ((nabs.T @ qualities)/len(rewards)).squeeze()
            if self.v is None:
                self.v = np.zeros_like(pol_grad)
            if self.m is None:
                self.m = np.zeros_like(pol_grad)

            self.model.update_params(pol_grad,self.lr)
        self.buffer.reset()
        return pol_grad




    def update_buffers(self,reward,action,state,new_state,nab,t_nab):
        self.buffer.rewards.append(reward)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        self.buffer.nxt_states.append(new_state)
        self.buffer.nabs.append(nab)
        self.buffer.t_nabs.append(t_nab)
        # self.buffer.nabs.append(self.nab)
        # # Make sure we dont use it twice :)
        # del self.nab



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def adam_update_single(grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Calculates the Adam-adjusted gradient update for a single set of parameters.

    Args:
        grads (numpy array): The gradients of the parameters.
        m (numpy array): Exponentially weighted average of past gradients (momentum).
                         Should be initialized as a zero array with the same shape as grads.
        v (numpy array): Exponentially weighted average of squared past gradients.
                         Should be initialized as a zero array with the same shape as grads.
        t (int): Timestep (should start from 1 and be incremented in the training loop).
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small scalar to avoid division by zero. Defaults to 1e-8.

    Returns:
        tuple:
            - adjusted_grad_update (numpy array): The Adam-adjusted gradient update term.
                                                 (This is learning_rate * m_corrected / (sqrt(v_corrected) + epsilon))
            - updated_m (numpy array): Updated first moment estimate.
            - updated_v (numpy array): Updated second moment estimate.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [grads, m, v]):
        raise TypeError("grads, m, and v must be NumPy arrays.")
    if not (grads.shape == m.shape == v.shape):
        raise ValueError("grads, m, and v must have the same shape.")
    if t < 1:
        raise ValueError("Timestep 't' must be 1 or greater for bias correction.")

    # Update biased first moment estimate
    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_updated = beta1 * m + (1 - beta1) * grads

    # Update biased second raw moment estimate
    # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
    v_updated = beta2 * v + (1 - beta2) * (grads ** 2)

    # Compute bias-corrected first moment estimate
    # m_hat_t = m_t / (1 - beta1^t)
    # The power ** t is applied to beta1 and beta2, not the entire denominator.
    m_corrected = m_updated / (1 - beta1 ** t)

    # Compute bias-corrected second raw moment estimate
    # v_hat_t = v_t / (1 - beta2^t)
    v_corrected = v_updated / (1 - beta2 ** t)

    # Calculate the Adam-adjusted gradient update
    # This is the term that will be subtracted from (or added to) the parameters.
    adjusted_grad_update = learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)

    return adjusted_grad_update, m_updated, v_updated



class ExperienceBuffer:
    def __init__(self):
        self.rewards = []
        self.actions= []
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
