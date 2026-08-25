
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.sparse import block_diag, csr_matrix, hstack, identity, lil_matrix, vstack
from tqdm import tqdm

import wandb
from src import actor
from src.critic import gae
from src.gym_envs import example_env, portfolio_env
from src.models import example_model, portfolio_model
from src.ppo.ppo_utils import PPO_MILP_Agent, PPOBuffer, PPOStep
from src.solvers import bnb, scip, scip_brute


def build_solver(config):
    training_cfg = config.get("training", {})
    solver_name = str(training_cfg.get("solver", "scip")).strip().lower()
    scip_cfg = config.get("scip", {})
    scip_brute_cfg = config.get("scip_brute", {})

    if solver_name == "bnb":
        return bnb.BranchAndBoundRevamped()

    if solver_name == "scip":
        use_standard_solver = bool(scip_cfg.get("use_standard_solver", True))

        if use_standard_solver:
            return scip.SCIPSolver(verbose=bool(scip_cfg.get("verbose", False)))

        return scip_brute.SCIPSolver(
            verbose=bool(scip_brute_cfg.get("verbose", False)),
            disable_heuristics=bool(scip_brute_cfg.get("disable_heuristics", True)),
            disable_presolve=bool(scip_brute_cfg.get("disable_presolve", True)),
            disable_separating=bool(scip_brute_cfg.get("disable_separating", True)),
            disable_propagation=bool(scip_brute_cfg.get("disable_propagation", False)),
            disable_conflict_analysis=bool(scip_brute_cfg.get("disable_conflict_analysis", False)),
            disable_symmetry=bool(scip_brute_cfg.get("disable_symmetry", False)),
            prefer_most_fractional_branching=bool(scip_brute_cfg.get("prefer_most_fractional_branching", False)),
            prefer_breadth_first=bool(scip_brute_cfg.get("prefer_breadth_first", False)),
            tighten_integer_projected_bounds=bool(scip_brute_cfg.get("tighten_integer_projected_bounds", False)),
            mimic_bnb_pool_filter=bool(scip_brute_cfg.get("mimic_bnb_pool_filter", False)),
            prefer_depth_first=bool(scip_brute_cfg.get("prefer_depth_first", True)),
        )

    raise ValueError(
        "training.solver must be either 'bnb' or 'scip'. "
        f"Got '{solver_name}'."
    )


def resolve_runtime_device(configured_device):
    device = str(configured_device).strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _safe_mean(total, count):
    return total / count if count else 0.0


def _empirical_cvar(losses, alpha):
    losses = np.asarray(losses, dtype=float).flatten()
    if losses.size == 0:
        return 0.0
    alpha = float(np.clip(alpha, 0.0, 0.999999))
    tail_count = max(1, int(np.ceil((1.0 - alpha) * losses.size)))
    sorted_losses = np.sort(losses)
    tail_losses = sorted_losses[-tail_count:]
    return float(np.mean(tail_losses))


def _print_run_diagnostics(problem_name, step, diagnostics):
    return None


def sanitize_env_action(env, action):
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    action_arr = np.round(action_arr).astype(np.int32)
    if hasattr(env.action_space, "nvec"):
        nvec = env.action_space.nvec.astype(np.int32)
        action_arr = np.clip(action_arr, 0, nvec - 1)
    return action_arr


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - np.max(z)
    ez = np.exp(z)
    s = np.sum(ez)
    if s <= 0:
        return np.ones_like(z, dtype=np.float64) / max(len(z), 1)
    return ez / s


def compute_ppo_linearization_stats(recent_ppo_samples, theta_now):
    if len(recent_ppo_samples) == 0:
        return {}

    mae_vals = []
    max_abs_vals = []
    chosen_shift_vals = []
    argmin_match_vals = []
    theta_delta_norm_vals = []
    chosen_grad_norm_vals = []
    pool_grad_norm_mean_vals = []
    grad_parallelism_vals = []
    policy_shift_std_vals = []

    for sample in recent_ppo_samples:
        obj_vals = sample["obj_vals"]
        theta_grads = sample["theta_grads"]
        theta_ref = sample["theta_ref"]
        action_idx = sample["action_idx"]

        theta_delta = theta_now - theta_ref
        obj_lin = obj_vals + theta_grads @ theta_delta

        diff = obj_lin - obj_vals
        mae_vals.append(float(np.mean(np.abs(diff))))
        max_abs_vals.append(float(np.max(np.abs(diff))))
        chosen_shift_vals.append(float(diff[action_idx]))
        argmin_match_vals.append(float(np.argmin(obj_lin) == np.argmin(obj_vals)))
        theta_delta_norm_vals.append(float(np.linalg.norm(theta_delta)))

        grad_norms = np.linalg.norm(theta_grads, axis=1)
        chosen_grad_norm_vals.append(float(grad_norms[action_idx]))
        pool_grad_norm_mean_vals.append(float(np.mean(grad_norms)))

        # Detect softmax invariance: measure if all gradients are parallel
        grad_mean = np.mean(theta_grads, axis=0, keepdims=True)
        grad_mean_norm = np.linalg.norm(grad_mean)
        if grad_mean_norm > 1e-8:
            proj = (theta_delta @ grad_mean.T) / (grad_mean_norm ** 2)
            grad_parallelism = float(proj[0] * grad_mean_norm / (np.linalg.norm(theta_delta) + 1e-8))
        else:
            grad_parallelism = 0.0
        grad_parallelism_vals.append(grad_parallelism)

        # Measure policy shift diversity
        policy_shifts = obj_lin - obj_vals
        policy_shift_std = float(np.std(policy_shifts))
        policy_shift_std_vals.append(policy_shift_std)

    return {
        "lin_obj_mae": float(np.mean(mae_vals)),
        "lin_obj_max_abs": float(np.max(max_abs_vals)),
        "lin_obj_chosen_shift_mean": float(np.mean(chosen_shift_vals)),
        "lin_obj_argmin_match": float(np.mean(argmin_match_vals)),
        "theta_delta_ref_norm_mean": float(np.mean(theta_delta_norm_vals)),
        "theta_delta_ref_norm_max": float(np.max(theta_delta_norm_vals)),
        "chosen_theta_grad_norm_mean": float(np.mean(chosen_grad_norm_vals)),
        "pool_theta_grad_norm_mean": float(np.mean(pool_grad_norm_mean_vals)),
        "grad_parallelism_mean": float(np.mean(grad_parallelism_vals)),
        "policy_shift_std_mean": float(np.mean(policy_shift_std_vals)),
    }


def compute_ppo_exactness_stats(recent_ppo_samples, ppo_agent, model, solver, max_samples=8):
    if len(recent_ppo_samples) == 0:
        return {}

    max_samples = int(max(1, max_samples))
    samples = recent_ppo_samples[-max_samples:]

    argmin_action_match_vals = []
    best_obj_gap_vals = []
    logp_abs_err_vals = []
    ratio_lin_vals = []
    ratio_exact_vals = []
    action_match_l2_vals = []

    # Make sure exact pools are solved under the current theta.
    ppo_agent._sync_model_params_from_theta()
    beta = float(ppo_agent.beta)

    for sample in samples:
        state = np.asarray(sample["state"], dtype=np.float32).reshape(-1)
        obj_vals = np.asarray(sample["obj_vals"], dtype=np.float32)
        theta_grads = np.asarray(sample["theta_grads"], dtype=np.float32)
        theta_ref = np.asarray(sample["theta_ref"], dtype=np.float32)
        action_idx = int(sample["action_idx"])
        old_logp = float(sample["old_logp"])
        lin_actions = np.asarray(sample["actions"], dtype=np.float32)

        if obj_vals.size == 0 or theta_grads.shape[0] == 0:
            continue

        theta_now = ppo_agent.theta.detach().cpu().numpy().astype(np.float32)
        theta_delta = theta_now - theta_ref
        obj_lin = obj_vals + theta_grads @ theta_delta
        p_lin = softmax_np(-beta * obj_lin)

        lin_argmin_idx = int(np.argmin(obj_lin))
        lin_argmin_action = lin_actions[lin_argmin_idx]

        model.update_state(state)
        node = model.get_LP_formulation()
        sol_pool_exact = solver.solve(node)
        if not sol_pool_exact:
            continue

        exact_obj = np.asarray([sol["fun"] for sol in sol_pool_exact], dtype=np.float32)
        exact_actions = np.asarray(
            [sol["x"][model.get_desc_var_indices()] for sol in sol_pool_exact],
            dtype=np.float32,
        )
        p_exact = softmax_np(-beta * exact_obj)

        exact_argmin_idx = int(np.argmin(exact_obj))
        exact_argmin_action = exact_actions[exact_argmin_idx]
        argmin_action_match_vals.append(float(np.allclose(lin_argmin_action, exact_argmin_action, atol=1e-6)))

        lin_best = float(np.min(obj_lin))
        exact_best = float(np.min(exact_obj))
        best_obj_gap_vals.append(abs(lin_best - exact_best))

        chosen_action = lin_actions[action_idx]
        dists = np.linalg.norm(exact_actions - chosen_action.reshape(1, -1), axis=1)
        exact_match_idx = int(np.argmin(dists))
        action_match_l2_vals.append(float(dists[exact_match_idx]))

        lin_prob = float(np.clip(p_lin[action_idx], 1e-12, 1.0))
        exact_prob = float(np.clip(p_exact[exact_match_idx], 1e-12, 1.0))
        logp_lin = float(np.log(lin_prob))
        logp_exact = float(np.log(exact_prob))
        logp_abs_err_vals.append(abs(logp_lin - logp_exact))

        ratio_lin_vals.append(float(np.exp(logp_lin - old_logp)))
        ratio_exact_vals.append(float(np.exp(logp_exact - old_logp)))

    n = len(argmin_action_match_vals)
    if n == 0:
        return {}

    ratio_lin_arr = np.asarray(ratio_lin_vals, dtype=np.float64)
    ratio_exact_arr = np.asarray(ratio_exact_vals, dtype=np.float64)
    if (
        ratio_lin_arr.size > 1
        and ratio_exact_arr.size > 1
        and np.std(ratio_lin_arr) > 1e-12
        and np.std(ratio_exact_arr) > 1e-12
    ):
        ratio_corr = float(np.corrcoef(ratio_lin_arr, ratio_exact_arr)[0, 1])
    else:
        ratio_corr = 0.0

    return {
        "n_samples": float(n),
        "argmin_action_match_rate": float(np.mean(argmin_action_match_vals)),
        "best_obj_abs_gap_mean": float(np.mean(best_obj_gap_vals)),
        "logp_abs_err_mean": float(np.mean(logp_abs_err_vals)),
        "ratio_abs_err_mean": float(np.mean(np.abs(ratio_lin_arr - ratio_exact_arr))),
        "ratio_corr": ratio_corr,
        "matched_action_l2_mean": float(np.mean(action_match_l2_vals)),
    }

def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.yaml"
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    runtime_device = resolve_runtime_device(config.get("device", "cpu"))

    problem_name = config.get("problem", "example")
    configured_seed = int(config["numpy_seed"])
    effective_seed = configured_seed
    portfolio_zero_init_seeds = set(config.get("portfolio_zero_init_seeds", [5, 6, 7]))
    force_portfolio_zero_init = problem_name == "portfolio" and configured_seed in portfolio_zero_init_seeds

    diagnostic_window = max(1, int(config.get("terminal_log_every", 1000)))

    # Init problem
    state_size = config["model"]["state_size"]
    action_size = config["model"]["action_size"]
    np.random.seed(effective_seed)
    num_cons = config["model"]["n_cons"]
    num_pieces = config["model"]["n_value_func"]
    aA = np.random.uniform(0,0.1,size = (num_pieces,state_size))
    aB = np.random.uniform(0,0.1,size = (num_pieces,action_size))
    b = np.random.uniform(0,.1,size=(num_pieces,))
    c = np.random.uniform(0,10,size=(action_size,))
    state = np.random.randint(2,size = state_size)


    C = np.random.uniform(0,1,size = (num_cons-2,state_size))
    C = np.vstack((C,np.zeros((2,state_size))))
    D = np.random.uniform(0,1,size = (num_cons,action_size))
    E = np.random.uniform(5,15,size = (num_cons-2))
    E2 = np.random.uniform(1,10,size = (2))
    E = np.hstack((E,E2))
    action_ub = 10
    
    bounds = [(0,action_ub) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    c_model = -np.random.uniform(0,10,size = (1,)) * np.ones((action_size,))
    A = np.random.uniform(0,.1,size = (state_size,state_size))
    B = np.random.uniform(0,1,size = (state_size,action_size))

    aA =np.vstack((aA,np.random.uniform(0,0.1,size = (5,state_size)))) 
                  
    aB =np.vstack((aB,np.random.uniform(0,0.1,size = (5,action_size))) )
    # # Init solver and gym model
    b = np.hstack((b,np.random.uniform(0,.1,size=(5,))))

    if force_portfolio_zero_init:
        # For selected portfolio seeds, use a neutral deterministic LP init.
        n_pieces_total = num_pieces + 5
        aA = np.zeros((n_pieces_total, state_size))
        aB = np.zeros((n_pieces_total, action_size))
        b = np.zeros((n_pieces_total,))
        C = np.zeros((num_cons, state_size))
        D = np.zeros((num_cons, action_size))
        E = np.full((num_cons,), 1e6)
        c_model = np.zeros((action_size,))

    load = config["load"]
    if load:
        load_path = Path(config["load_path"])
        if not load_path.is_absolute():
            load_path = project_root / load_path
        with load_path.open() as params_file:
            params = yaml.safe_load(params_file)
        aA = np.array(params['aA'])
        aB = np.array(params['aB'])
        c_model = np.array(params['c'])
        b = np.array(params['b'])
        
    if problem_name == "example":
        model_cls = example_model.Arbbin
        env_cls = example_env.Arb_binary
        model_kwargs = {}
        env_kwargs = {}
        init_env_seed = 0
    elif problem_name == "portfolio":
        model_cls = portfolio_model.PortfolioModel
        env_cls = portfolio_env.PortfolioEnv
        portfolio_cfg = config.get("portfolio_env", {})
        portfolio_model_cfg = config.get("portfolio_model", {})

        seed_behavior = "modern" if configured_seed >= 5 else "legacy"
        transaction_cost = portfolio_cfg.get("transaction_cost", 0.0)
        holding_cost = portfolio_cfg.get("holding_cost", 0.0)
        budget_cap = portfolio_cfg.get("budget_cap", None)
        initial_cash = portfolio_cfg.get("initial_cash", 30.0)
        cash_interest_rate = portfolio_cfg.get("cash_interest_rate", 0.0)
        risk_cap = portfolio_cfg.get("risk_cap", None)
        risk_weight = portfolio_cfg.get("risk_weight", 1.0)
        asset_max_position = portfolio_cfg.get("asset_max_position", None)
        market_mode = portfolio_cfg.get("market_mode", "linear")
        action_mode = portfolio_cfg.get("action_mode", "absolute")
        reward_mode = portfolio_cfg.get("reward_mode", "economic")
        inventory_penalty = portfolio_cfg.get("inventory_penalty", 0.0)
        return_mu = portfolio_cfg.get("return_mu", 0.0)
        return_phi = portfolio_cfg.get("return_phi", 0.0)
        return_sigma = portfolio_cfg.get("return_sigma", 0.01)
        alpha_mode = portfolio_cfg.get("alpha_mode", "off")
        alpha_rho = portfolio_cfg.get("alpha_rho", 0.9)
        alpha_sigma = portfolio_cfg.get("alpha_sigma", 0.02)
        alpha_to_return = portfolio_cfg.get("alpha_to_return", 0.2)
        signal_noise_std = portfolio_cfg.get("signal_noise_std", 0.01)
        return_signal_scale = portfolio_cfg.get("return_signal_scale", 1.0)
        cvar_mode = portfolio_cfg.get("cvar_mode", "off")
        cvar_cap = portfolio_cfg.get("cvar_cap", 1.0)
        cvar_alpha = portfolio_cfg.get("cvar_alpha", 0.95)
        cvar_n_scenarios = portfolio_cfg.get("cvar_n_scenarios", 20)
        cvar_obj_weight = portfolio_cfg.get("cvar_obj_weight", 0.0)
        price_levels_mode = portfolio_cfg.get("price_levels_mode", "off")
        initial_asset_price = portfolio_cfg.get("initial_asset_price", 100.0)
        min_asset_price = portfolio_cfg.get("min_asset_price", 1.0)

        lr_mult_aA = portfolio_model_cfg.get("lr_mult_aA", 1.0)
        lr_mult_aB = portfolio_model_cfg.get("lr_mult_aB", 1.0)
        lr_mult_b = portfolio_model_cfg.get("lr_mult_b", 1.0)
        position_dynamics_mode = portfolio_model_cfg.get("position_dynamics_mode", "legacy")

        env_kwargs = {
            "transaction_cost": transaction_cost,
            "holding_cost": holding_cost,
            "budget_cap": budget_cap,
            "initial_cash": initial_cash,
            "cash_interest_rate": cash_interest_rate,
            "risk_cap": risk_cap,
            "risk_weight": risk_weight,
            "asset_max_position": asset_max_position,
            "action_mode": action_mode,
            "reward_mode": reward_mode,
            "inventory_penalty": inventory_penalty,
            "market_mode": market_mode,
            "return_mu": return_mu,
            "return_phi": return_phi,
            "return_sigma": return_sigma,
            "alpha_mode": alpha_mode,
            "alpha_rho": alpha_rho,
            "alpha_sigma": alpha_sigma,
            "alpha_to_return": alpha_to_return,
            "signal_noise_std": signal_noise_std,
            "cvar_n_scenarios": cvar_n_scenarios,
            "cvar_alpha": cvar_alpha,
            "price_levels_mode": price_levels_mode,
            "initial_asset_price": initial_asset_price,
            "min_asset_price": min_asset_price,
            "seed_behavior": seed_behavior,
        }
        model_kwargs = {
            "transaction_cost": transaction_cost,
            "holding_cost": holding_cost,
            "budget_cap": budget_cap,
            "initial_cash": initial_cash,
            "risk_cap": risk_cap,
            "risk_weight": risk_weight,
            "asset_max_position": asset_max_position,
            "action_mode": action_mode,
            "market_mode": market_mode,
            "return_signal_scale": return_signal_scale,
            "lr_mult_aA": lr_mult_aA,
            "lr_mult_aB": lr_mult_aB,
            "lr_mult_b": lr_mult_b,
            "position_dynamics_mode": position_dynamics_mode,
            "cvar_mode": cvar_mode,
            "cvar_cap": cvar_cap,
            "cvar_alpha": cvar_alpha,
            "cvar_n_scenarios": cvar_n_scenarios,
            "cvar_obj_weight": cvar_obj_weight,
            "price_levels_mode": price_levels_mode,
            "initial_asset_price": initial_asset_price,
        }
        init_env_seed = 0 if force_portfolio_zero_init else (effective_seed if configured_seed >= 5 else 0)
    else:
        raise ValueError(f"Unsupported problem '{problem_name}'. Expected 'example' or 'portfolio'.")

    m = model_cls(
        c_model,
        C,
        D,
        E,
        aA,
        aB,
        b,
        bounds,
        integer,
        config["model"]["penalty_factor"],
        **model_kwargs,
    )

    gym_model = env_cls(
        c,
        np.zeros_like(state),
        A,
        B,
        C,
        D,
        E,
        config["gym"]["pf"],
        a_space_size=11,
        std=config["gym"]["noise_std"],
        **env_kwargs,
    )
    m.update_state(gym_model.reset(seed=init_env_seed)[0])
    if hasattr(m, "update_prev_action") and hasattr(gym_model, "prev_action"):
        m.update_prev_action(gym_model.prev_action)
    if hasattr(m, "update_cash") and hasattr(gym_model, "cash"):
        m.update_cash(gym_model.cash)
    if hasattr(m, "update_prices") and hasattr(gym_model, "prices"):
        m.update_prices(gym_model.prices)

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    run = wandb.init(name = config["name"],mode = config["wandb_mode"],config = config)


    window_size = config["plotting"]["window_size"]
    training_cfg = config.get("training", {})
    algorithm = training_cfg.get("algorithm", "vanilla_gradient").lower()
    solver_name = str(training_cfg.get("solver", "scip")).strip().lower()
    if algorithm not in ["vanilla_gradient", "ppo"]:
        raise ValueError(
            "training.algorithm must be either 'vanilla_gradient' or 'ppo'. "
            f"Got '{algorithm}'."
        )

    act_lr = config["actor"]["lr"]
    critic_lr = config["critic"]["lr"]
    df = config["critic"]["df"]
    beta = config["actor"]["beta"]
    eps = config["critic"]["eps"]

    n_actions = action_ub*(100+10+1)+1

    dims = [state_size,128,128,1]
    solver = build_solver(config)

    act = None
    ppo_agent = None
    if algorithm == "vanilla_gradient":
        critic = gae.GAE(dims,critic_lr,df,eps,0.1,runtime_device)
        act = actor.Actor(
            m,
            solver,
            critic,
            beta=beta,
            lr=act_lr,
            df=df,
            nn_sample=config["actor"]["nn_sample"],
            sampled_grad=config["actor"]["sampled_grad"],
        )
    else:
        ppo_cfg = config.get("ppo", {})
        ppo_agent = PPO_MILP_Agent(
            model=m,
            solver=solver,
            state_dim=state_size,
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_param=float(ppo_cfg.get("clip_param", 0.2)),
            entropy_coef=float(ppo_cfg.get("entropy_coef", 0.01)),
            value_coef=float(ppo_cfg.get("value_coef", 0.5)),
            lr_policy=float(ppo_cfg.get("actor_lr", act_lr)),
            lr_value=float(ppo_cfg.get("critic_lr", critic_lr)),
            policy_beta=float(ppo_cfg.get("policy_beta", beta)),
            update_epochs=int(ppo_cfg.get("opt_epochs", config["train_iters"])),
            mini_batch_size=int(ppo_cfg.get("mini_batch_size", 64)),
            target_kl=float(ppo_cfg.get("target_kl", 0.0)),
            normalize_adv=bool(ppo_cfg.get("normalize_adv", True)),
            normalize_obj_values=bool(ppo_cfg.get("normalize_obj_values", True)),
            obj_norm_eps=float(ppo_cfg.get("obj_norm_eps", 1e-8)),
            minimize_env_reward=bool(ppo_cfg.get("minimize_env_reward", True)),
            normalize_rewards=bool(ppo_cfg.get("normalize_rewards", True)),
            reward_norm_eps=float(ppo_cfg.get("reward_norm_eps", 1e-8)),
            reward_clip=ppo_cfg.get("reward_clip", None),
            nn_sample=bool(ppo_cfg.get("nn_sample", config.get("actor", {}).get("nn_sample", True))),
            device=runtime_device,
        )

    state = gym_model.state



    training_iters = config["train_iters"]
    vanilla_rollout_iters = config["rollout_iters"]
    ppo_rollout_iters = int(config.get("ppo", {}).get("rollout_iters", vanilla_rollout_iters))
    rollout_iters = ppo_rollout_iters if algorithm == "ppo" else vanilla_rollout_iters
    total_iters = config["total_iters"]


    ep_reward =0
    economic_ep_reward = 0
    ep_rewards = []
    rewards = []
    diagnostics = {
        "step_count": 0,
        "reward_sum": 0.0,
        "economic_reward_sum": 0.0,
        "reward_sq_sum": 0.0,
        "episode_count": 0,
        "episode_reward_sum": 0.0,
        "economic_episode_reward_sum": 0.0,
        "episode_length_sum": 0.0,
        "no_action_count": 0,
        "turnover_sum": 0.0,
        "risk_utilization_sum": 0.0,
        "empirical_cvar_sum": 0.0,
    }


    T = config["explicit_sol_time"]
    fathomed_counter = 0
    ep_length = 0

    comp_expected = config["comp_expected"]
    comp_expected_every = config["comp_expected_every"]

    recent_rewards = []
    recent_n_sols = []
    recent_ppo_samples = []
    recent_action_mismatch = []
    recent_action_mismatch_l2 = []

    columns = ["c1", "c2", "c3"]
    c_table = wandb.Table(columns=columns)

    iter_counter = 0
    expected_ep_reward = None
    last_calced = 0
    for _ in tqdm(range(total_iters),desc= "Total Iterations"):
        ppo_buffer = PPOBuffer() if algorithm == "ppo" else None

        if  last_calced > comp_expected_every and comp_expected:
            expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)
            last_calced = 0

        for i in tqdm(range(rollout_iters),leave=False,desc= "Rollout"):
            iter_counter += 1
            ep_length += 1
            last_calced +=1
            diagnostics["step_count"] += 1

            if hasattr(m, "update_prev_action") and hasattr(gym_model, "prev_action"):
                m.update_prev_action(gym_model.prev_action)
            if hasattr(m, "update_cash") and hasattr(gym_model, "cash"):
                m.update_cash(gym_model.cash)
            if hasattr(m, "update_prices") and hasattr(gym_model, "prices"):
                m.update_prices(gym_model.prices)
            if hasattr(m, "update_scenarios") and hasattr(gym_model, "return_window"):
                m.update_scenarios(gym_model.return_window)


            store = True
            if algorithm == "vanilla_gradient":
                act_out = act.act(state)
                if act_out is None:
                    action, act_info = None, {"fathomed": False, "nab": 0.0, "t_nab": 0.0, "n_sols": 0}
                else:
                    action, act_info = act_out
                fathomed_counter =  fathomed_counter + 1 if act_info["fathomed"] else fathomed_counter
                nab = act_info["nab"]
                t_nab = act_info["t_nab"]
                if action is None:
                    action = np.zeros_like(state)
                    store = False
                    diagnostics["no_action_count"] += 1
                action = sanitize_env_action(gym_model, action)
            else:
                chosen_action_raw = None
                action, act_info = ppo_agent.act(state)
                if action is None or act_info is None:
                    action = np.zeros((action_size,), dtype=np.int32)
                    store = False
                    diagnostics["no_action_count"] += 1
                else:
                    chosen_action_raw = np.asarray(action, dtype=np.float32).reshape(-1)
                    action = sanitize_env_action(gym_model, action)

                if store and chosen_action_raw is not None:
                    executed_action = np.asarray(action, dtype=np.float32).reshape(-1)
                    mismatch = float(not np.allclose(chosen_action_raw, executed_action, atol=1e-6))
                    mismatch_l2 = float(np.linalg.norm(chosen_action_raw - executed_action))
                    recent_action_mismatch.append(mismatch)
                    recent_action_mismatch_l2.append(mismatch_l2)
                    if len(recent_action_mismatch) > diagnostic_window:
                        recent_action_mismatch = recent_action_mismatch[-diagnostic_window:]
                    if len(recent_action_mismatch_l2) > diagnostic_window:
                        recent_action_mismatch_l2 = recent_action_mismatch_l2[-diagnostic_window:]

            old_state_for_buffer = np.asarray(state).copy()
            state,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            turnover = float(info.get("turnover", 0.0))
            risk_utilization = float(info.get("risk_utilization", 0.0))
            economic_reward = float(info.get("economic_reward", -reward))
            empirical_cvar = 0.0
            scenario_matrix = info.get("scenario_matrix", None)
            target_position = np.asarray(info.get("target_position", np.zeros((action_size,))), dtype=float)
            if scenario_matrix is not None and hasattr(m, "cvar_mode") and m.cvar_mode == "on":
                scenario_matrix = np.asarray(scenario_matrix, dtype=float)
                if scenario_matrix.ndim == 2 and scenario_matrix.shape[1] == target_position.shape[0]:
                    scenario_losses = -(scenario_matrix @ target_position)
                    empirical_cvar = _empirical_cvar(scenario_losses, getattr(m, "cvar_alpha", 0.95))

            diagnostics["turnover_sum"] += turnover
            diagnostics["risk_utilization_sum"] += risk_utilization
            diagnostics["empirical_cvar_sum"] += empirical_cvar
            
            if algorithm == "vanilla_gradient" and store:
                act.update_buffers(reward,action_number,old_state,new_state,nab,t_nab)
            elif algorithm == "ppo" and store:
                ppo_buffer.add(
                    PPOStep(
                        state=np.asarray(old_state_for_buffer, dtype=np.float32),
                        obj_vals=np.asarray(act_info["obj_vals"], dtype=np.float32),
                        theta_grads=np.asarray(act_info["theta_grads"], dtype=np.float32),
                        theta_ref=np.asarray(act_info["theta_ref"], dtype=np.float32),
                        action_idx=int(act_info["action_idx"]),
                        reward=float(reward),
                        done=bool(terminated),
                        old_logp=float(act_info["old_logp"]),
                        value=float(act_info["value"]),
                        chosen_action_raw=np.asarray(chosen_action_raw, dtype=np.float32),
                        executed_action=np.asarray(action, dtype=np.float32),
                    )
                )
                recent_ppo_samples.append(
                    {
                        "state": np.asarray(old_state_for_buffer, dtype=np.float32),
                        "obj_vals": np.asarray(act_info["obj_vals"], dtype=np.float32),
                        "actions": np.asarray(act_info["actions"], dtype=np.float32),
                        "theta_grads": np.asarray(act_info["theta_grads"], dtype=np.float32),
                        "theta_ref": np.asarray(act_info["theta_ref"], dtype=np.float32),
                        "action_idx": int(act_info["action_idx"]),
                        "old_logp": float(act_info["old_logp"]),
                    }
                )
                if len(recent_ppo_samples) > diagnostic_window:
                    recent_ppo_samples = recent_ppo_samples[-diagnostic_window:]

            ep_reward += reward
            economic_ep_reward += economic_reward
            n_sols = 0 if (act_info is None) else act_info.get("n_sols", 0)
            diagnostics["reward_sum"] += reward
            diagnostics["economic_reward_sum"] += economic_reward
            diagnostics["reward_sq_sum"] += reward ** 2
            recent_rewards.append(float(reward))
            recent_n_sols.append(float(n_sols))
            if len(recent_rewards) > diagnostic_window:
                recent_rewards = recent_rewards[-diagnostic_window:]
            if len(recent_n_sols) > diagnostic_window:
                recent_n_sols = recent_n_sols[-diagnostic_window:]
            run.log({
                "reward" : reward,
                "economic_reward": economic_reward,
                "action" : action_number,
                "n_sols" : n_sols,
                "turnover": turnover,
                "risk_utilization": risk_utilization,
                "empirical_cvar": empirical_cvar,
            })

            if terminated or i == rollout_iters-1:
                ep_rewards.append(ep_reward)
                diagnostics["episode_count"] += 1
                diagnostics["episode_reward_sum"] += ep_reward
                diagnostics["economic_episode_reward_sum"] += economic_ep_reward
                diagnostics["episode_length_sum"] += ep_length

                metric = {
                    "ep_reward" : ep_reward,
                    "economic_ep_reward": economic_ep_reward,
                    "fathomed_counter" : fathomed_counter,
                    "ep_length" : ep_length
                          }
                
                if len(ep_rewards) == window_size:
                    metric["smooth_ep_reward"] = sum(ep_rewards)/window_size
                    ep_rewards = []
                if comp_expected and expected_ep_reward is not None:
                    metric["expected_ep_reward"] = expected_ep_reward
                    metric["distance_from_opt_pol"] = expected_ep_reward - ep_reward
                    expected_ep_reward = None
                run.log(metric)
                ep_reward = 0
                economic_ep_reward = 0
                fathomed_counter = 0
                ep_length = 0
                state,_ = gym_model.reset()
                if hasattr(m, "update_prev_action") and hasattr(gym_model, "prev_action"):
                    m.update_prev_action(gym_model.prev_action)
                if hasattr(m, "update_cash") and hasattr(gym_model, "cash"):
                    m.update_cash(gym_model.cash)
                if hasattr(m, "update_prices") and hasattr(gym_model, "prices"):
                    m.update_prices(gym_model.prices)
                if last_calced > comp_expected_every and comp_expected:
                    expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)
                    last_calced = 0


        if algorithm == "vanilla_gradient":
            pol_grad = act.train(iters = training_iters,sample = config["actor"]["sample"],num_samples=config["actor"]["num_samples"])
            pol_grad_norm = np.linalg.norm(pol_grad)
            ppo_metrics = {}
            ppo_invariance_stats = {}
        else:
            bootstrap_value = 0.0
            if len(ppo_buffer) > 0 and not ppo_buffer.steps[-1].done:
                state_final = torch.as_tensor(
                    np.asarray(state, dtype=np.float32).reshape(-1),
                    dtype=torch.float32,
                    device=ppo_agent.device,
                )
                with torch.no_grad():
                    bootstrap_value = float(
                        ppo_agent.value_net(state_final.unsqueeze(0)).squeeze(0).item()
                    )
            ppo_metrics = ppo_agent.update(ppo_buffer, last_value=bootstrap_value) if len(ppo_buffer) > 0 else {}
            pol_grad_norm = ppo_metrics.get("theta_norm", 0.0)
            
            # Compute invariance diagnostics AFTER update using stored samples
            ppo_invariance_stats = {}
            if len(recent_ppo_samples) > 0:
                theta_now = ppo_agent.theta.detach().cpu().numpy().astype(np.float32)
                ppo_invariance_stats = compute_ppo_linearization_stats(recent_ppo_samples, theta_now)

        c_diff = ((-c -m.c )**2).mean()
        aA_change = np.sum((aA-m.aA)**2 )
        aB_change = np.sum((aB-m.aB)**2)
        b_change = np.sum((b-m.b)**2)
        metrics = {"c_diff" : c_diff , "aA_change" : aA_change , "aB_change" : aB_change, "b_change" : b_change,"pol_grad": pol_grad_norm}
        for k, v in ppo_metrics.items():
            metrics[f"ppo_{k}"] = v
        run.log(metrics)


def formulate_lp_with_initial_state(c, A, B, D, E, F, T, s_initial):
    """
    Formulates the time-dependent LP into standard form min c'z s.t.
    A_eq z = b_eq, A_ub z <= b_ub, assuming a fixed initial state s_0.

    Args:
        c (np.ndarray): Cost vector for x_t (dim n).
        A (np.ndarray or sparse matrix): State transition matrix for s_t (dim m x m).
        B (np.ndarray or sparse matrix): State transition matrix for x_t (dim m x n).
        D (np.ndarray or sparse matrix): Inequality matrix for s_t (dim k x m).
        E (np.ndarray or sparse matrix): Inequality matrix for x_t (dim k x n).
        F (np.ndarray): Right-hand side for inequality constraints (dim k).
        T (int): Time horizon (number of steps, x_t goes from 0 to T-1).
        s_initial (np.ndarray): The fixed initial state vector s_0 (dim m).

    Returns:
        tuple: (c_agg, A_eq, b_eq, A_ub, b_ub)
               Ready for scipy.optimize.linprog (bounds need to be added separately).
               Matrices A_eq and A_ub are returned as CSR sparse matrices.
    """
    # Ensure inputs are numpy arrays for shape info
    c = np.asarray(c)
    s_initial = np.asarray(s_initial)
    F = np.asarray(F)

    # --- Dimensions ---
    n = B.shape[1]  # Dimension of x_t
    m = A.shape[0]  # Dimension of s_t
    if D is not None and E is not None:
       k = D.shape[0] # Number of inequality constraints per step
    else: # Handle case with no inequality constraints D, E, F
        k = 0


    if s_initial.shape[0] != m:
        raise ValueError(f"s_initial dimension ({s_initial.shape[0]}) must match A rows ({m})")
    if c.shape[0] != n:
        raise ValueError(f"c dimension ({c.shape[0]}) must match B columns ({n})")
    if k > 0 and F.shape[0] != k:
         raise ValueError(f"F dimension ({F.shape[0]}) must match D rows ({k})")


    N = T * n + (T + 1) * m # Total number of variables in z

    # --- Aggregated Cost Vector c_agg ---
    c_agg_x = np.tile(c, T)
    c_agg_s = np.zeros((T + 1) * m)
    c_agg = np.hstack([c_agg_x, c_agg_s])

    # --- Equality Constraints (Dynamics) A_eq_dynamics z = 0 ---
    num_eq_dynamics = T * m
    A_eq_dynamics = lil_matrix((num_eq_dynamics, N))
    I_m = identity(m, format='csr') # Use sparse identity

    for t in range(T):
        row_start = t * m
        row_end = (t + 1) * m

        col_start_xt = t * n
        col_end_xt = (t + 1) * n

        col_start_st = T * n + t * m
        col_end_st = T * n + (t + 1) * m

        col_start_st1 = T * n + (t + 1) * m
        col_end_st1 = T * n + (t + 2) * m

        A_eq_dynamics[row_start:row_end, col_start_xt:col_end_xt] = B
        A_eq_dynamics[row_start:row_end, col_start_st:col_end_st] = A
        A_eq_dynamics[row_start:row_end, col_start_st1:col_end_st1] = -I_m

    b_eq_dynamics = np.zeros(num_eq_dynamics)

    # --- Equality Constraints (Initial State) A_eq_s0 z = s_initial ---
    num_eq_s0 = m
    A_eq_s0 = lil_matrix((num_eq_s0, N))
    s0_col_start = T * n # Column index where s_0 variables begin
    s0_col_end = T * n + m
    A_eq_s0[:, s0_col_start:s0_col_end] = I_m

    b_eq_s0 = s_initial # RHS is the fixed initial state

    # --- Combine Equality Constraints ---
    A_eq = vstack([A_eq_dynamics, A_eq_s0], format='csr')
    b_eq = np.concatenate([b_eq_dynamics, b_eq_s0])

    # --- Inequality Constraints A_ub z <= b_ub ---
    if k > 0:
        num_ineq = T * k
        A_ub_x = block_diag([E] * T, format='csr') # Size (T*k) x (T*n)
        A_ub_s_main = block_diag([D] * T, format='csr') # Size (T*k) x (T*m) (for s_0 to s_{T-1})
        A_ub_s_T_zeros = csr_matrix((num_ineq, m)) # Zero block for s_T columns, Size (T*k) x m
        A_ub_s = hstack([A_ub_s_main, A_ub_s_T_zeros], format='csr') # Size (T*k) x ((T+1)*m)
        A_ub = hstack([A_ub_x, A_ub_s], format='csr') # Size (T*k) x N

        if F.ndim > 1:
            F_flat = F.flatten()
        else:
            F_flat = F
        b_ub = np.tile(F_flat, T)
    else: # No inequality constraints
        # Create empty structures as placeholders or handle as needed by solver
        A_ub = None # Or csr_matrix((0, N)) depending on solver needs
        b_ub = None # Or np.array([])

    return c_agg, A_eq, b_eq, A_ub, b_ub



def calc_expected_reward(c,A,B,C,D,E,T,state,solver):
    sols = None
    while sols is None and T > 0:

        c_agg,A_eq,b_eq,A_ub,b_ub = formulate_lp_with_initial_state(c,A,B,C,D,E,T,state)

        bounds_agg = [(0,8) for _ in range(A_ub.shape[1])]
        integer_actions = [ 1 for _ in range(c.size*T)]
        integer_states = [ 0 for _ in range(A_ub.shape[1] - len(integer_actions))]
        integer_agg = integer_actions + integer_states
        node = {
            "c" : c_agg,
            "A_ub" : A_ub,
            "b_ub" : b_ub,
            "A_eq" : A_eq,
            "b_eq" : b_eq,
            "bounds" : bounds_agg,
            "integer" : integer_agg,
        }
            
        sols = solver.solve(node)
        if sols is None:
            T -= 1
        else:
            best = sols[0]
            for sol in sols:
                if sol["fun"] < best["fun"]:
                    best = sol
            return best["fun"]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('cumtime').print_stats(20)
    # # cProfile.run("main()k",sort = "time")
