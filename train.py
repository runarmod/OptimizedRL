import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import wandb
from src.config.config_loader import load_config
from src.config.config_models import AppConfig
from src.gym_envs import example_env, portfolio_env
from src.gym_envs.env_interface import Env
from src.models import example_model, portfolio_model
from src.models.model_interface import Model
from src.solvers import bnb, scip, scip_brute
from src.solvers.solver_interface import Solver
from src.training_algorithm.ppo.ppo import PPO_MILP_Agent
from src.training_algorithm.training_algorithm_interface import (
    AlgorithmTransition,
    TrainingAlgorithm,
)
from src.training_algorithm.vanilla_gradient.vanilla_gradient import (
    VanillaGradientAlgorithm,
)
from src.utils.calc_expected import calc_expected_reward


def build_training_algorithm(
    algorithm: str,
    config: AppConfig,
    model: Model,
    solver: Solver,
    runtime_device: str,
) -> TrainingAlgorithm:

    match algorithm:
        case "vanilla_gradient":
            return VanillaGradientAlgorithm(
                config=config,
                model=model,
                solver=solver,
                runtime_device=runtime_device,
            )

        case "ppo":
            ppo_cfg = config.ppo
            return PPO_MILP_Agent(
                model=model,
                solver=solver,
                state_dim=config.model.state_size,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
                clip_param=ppo_cfg.clip_param,
                entropy_coef=ppo_cfg.entropy_coef,
                value_coef=ppo_cfg.value_coef,
                lr_policy=ppo_cfg.actor_lr,
                lr_value=ppo_cfg.critic_lr,
                policy_beta=ppo_cfg.policy_beta,
                update_epochs=ppo_cfg.opt_epochs,
                mini_batch_size=ppo_cfg.mini_batch_size,
                target_kl=ppo_cfg.target_kl,
                normalize_adv=ppo_cfg.normalize_adv,
                normalize_obj_values=ppo_cfg.normalize_obj_values,
                obj_norm_eps=ppo_cfg.obj_norm_eps,
                minimize_env_reward=ppo_cfg.minimize_env_reward,
                normalize_rewards=ppo_cfg.normalize_rewards,
                reward_norm_eps=ppo_cfg.reward_norm_eps,
                reward_clip=ppo_cfg.reward_clip,
                nn_sample=ppo_cfg.nn_sample,
                device=runtime_device,
                rollout_iters=ppo_cfg.rollout_iters,
                diagnostic_window=config.terminal_log_every,
            )

    raise ValueError(
        f"training.algorithm must be 'vanilla_gradient' or 'ppo' " f"Got '{algorithm}'."
    )


def build_solver(config: AppConfig):
    match config.training.solver:
        case "bnb":
            return bnb.BranchAndBoundRevamped()

        case "scip":
            return scip.SCIPSolver(verbose=config.scip.verbose)

        case "scip_brute":
            return scip_brute.SCIPSolver(
                verbose=config.scip_brute.verbose,
                disable_heuristics=config.scip_brute.disable_heuristics,
                disable_presolve=config.scip_brute.disable_presolve,
                disable_separating=config.scip_brute.disable_separating,
                disable_propagation=config.scip_brute.disable_propagation,
                disable_conflict_analysis=config.scip_brute.disable_conflict_analysis,
                disable_symmetry=config.scip_brute.disable_symmetry,
                prefer_most_fractional_branching=config.scip_brute.prefer_most_fractional_branching,
                prefer_breadth_first=config.scip_brute.prefer_breadth_first,
                tighten_integer_projected_bounds=config.scip_brute.tighten_integer_projected_bounds,
                mimic_bnb_pool_filter=config.scip_brute.mimic_bnb_pool_filter,
                prefer_depth_first=config.scip_brute.prefer_depth_first,
            )

    raise ValueError(
        "training.solver must be one of 'scip', 'scip_brute', or 'bnb'. "
        f"Got '{config.training.solver}'."
    )


def resolve_runtime_device(configured_device):
    device = str(configured_device).strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def build_models(config: AppConfig, project_root: Path) -> tuple[Model, Env]:
    problem_name = config.problem
    configured_seed = config.numpy_seed
    effective_seed = configured_seed
    portfolio_zero_init_seeds = set(config.portfolio_zero_init_seeds)
    force_portfolio_zero_init = (
        problem_name == "portfolio" and configured_seed in portfolio_zero_init_seeds
    )

    state_size = config.model.state_size
    action_size = config.model.action_size
    np.random.seed(effective_seed)
    num_cons = config.model.n_cons
    num_pieces = config.model.n_value_func
    aA = np.random.uniform(0, 0.1, size=(num_pieces, state_size))
    aB = np.random.uniform(0, 0.1, size=(num_pieces, action_size))
    b = np.random.uniform(0, 0.1, size=(num_pieces,))
    c = np.random.uniform(0, 10, size=(action_size,))
    state = np.random.randint(2, size=state_size)

    C = np.random.uniform(0, 1, size=(num_cons - 2, state_size))
    C = np.vstack((C, np.zeros((2, state_size))))
    D = np.random.uniform(0, 1, size=(num_cons, action_size))
    E = np.random.uniform(5, 15, size=(num_cons - 2))
    E2 = np.random.uniform(1, 10, size=(2))
    E = np.hstack((E, E2))
    action_ub = 10

    bounds = [(0, action_ub) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    c_model = -np.random.uniform(0, 10, size=(1,)) * np.ones((action_size,))
    A = np.random.uniform(0, 0.1, size=(state_size, state_size))
    B = np.random.uniform(0, 1, size=(state_size, action_size))
    aA = np.vstack((aA, np.random.uniform(0, 0.1, size=(5, state_size))))
    aB = np.vstack((aB, np.random.uniform(0, 0.1, size=(5, action_size))))
    b = np.hstack((b, np.random.uniform(0, 0.1, size=(5,))))

    if force_portfolio_zero_init:
        n_pieces_total = num_pieces + 5
        aA = np.zeros((n_pieces_total, state_size))
        aB = np.zeros((n_pieces_total, action_size))
        b = np.zeros((n_pieces_total,))
        C = np.zeros((num_cons, state_size))
        D = np.zeros((num_cons, action_size))
        E = np.full((num_cons,), 1e6)
        c_model = np.zeros((action_size,))

    if config.load:
        load_path = config.load_path
        if load_path is None:
            raise ValueError("Config 'load' is True but 'load_path' is not set.")
        if not load_path.is_absolute():
            load_path = project_root / load_path
        with load_path.open() as params_file:
            params = yaml.safe_load(params_file)
        aA = np.array(params["aA"])
        aB = np.array(params["aB"])
        c_model = np.array(params["c"])
        b = np.array(params["b"])

    if problem_name == "example":
        model = example_model.Arbbin(
            c_model,
            C,
            D,
            E,
            aA,
            aB,
            b,
            bounds,
            integer,
            config.model.penalty_factor,
        )
        environment = example_env.Arb_binary(
            c,
            np.zeros_like(state),
            A,
            B,
            C,
            D,
            E,
            config.gym.pf,
            a_space_size=11,
            std=config.gym.noise_std,
        )
        init_env_seed = 0
    elif problem_name == "portfolio":
        portfolio_cfg = config.portfolio_env
        portfolio_model_cfg = config.portfolio_model
        model = portfolio_model.PortfolioModel(
            c_model,
            C,
            D,
            E,
            aA,
            aB,
            b,
            bounds,
            integer,
            config.model.penalty_factor,
            transaction_cost=portfolio_cfg.transaction_cost,
            holding_cost=portfolio_cfg.holding_cost,
            budget_cap=portfolio_cfg.budget_cap,
            initial_cash=portfolio_cfg.initial_cash,
            risk_cap=portfolio_cfg.risk_cap,
            risk_weight=portfolio_cfg.risk_weight,
            asset_max_position=portfolio_cfg.asset_max_position,
            action_mode=portfolio_cfg.action_mode,
            market_mode=portfolio_cfg.market_mode,
            return_signal_scale=portfolio_cfg.return_signal_scale,
            lr_mult_aA=portfolio_model_cfg.lr_mult_aA,
            lr_mult_aB=portfolio_model_cfg.lr_mult_aB,
            lr_mult_b=portfolio_model_cfg.lr_mult_b,
            position_dynamics_mode=portfolio_model_cfg.position_dynamics_mode,
            cvar_mode=portfolio_cfg.cvar_mode,
            cvar_cap=portfolio_cfg.cvar_cap,
            cvar_alpha=portfolio_cfg.cvar_alpha,
            cvar_n_scenarios=portfolio_cfg.cvar_n_scenarios,
            cvar_obj_weight=portfolio_cfg.cvar_obj_weight,
            price_levels_mode=portfolio_cfg.price_levels_mode,
            initial_asset_price=portfolio_cfg.initial_asset_price,
        )
        environment = portfolio_env.PortfolioEnv(
            c,
            np.zeros_like(state),
            A,
            B,
            C,
            D,
            E,
            config.gym.pf,
            a_space_size=11,
            std=config.gym.noise_std,
            transaction_cost=portfolio_cfg.transaction_cost,
            holding_cost=portfolio_cfg.holding_cost,
            budget_cap=portfolio_cfg.budget_cap,
            initial_cash=portfolio_cfg.initial_cash,
            cash_interest_rate=portfolio_cfg.cash_interest_rate,
            risk_cap=portfolio_cfg.risk_cap,
            risk_weight=portfolio_cfg.risk_weight,
            asset_max_position=portfolio_cfg.asset_max_position,
            action_mode=portfolio_cfg.action_mode,
            reward_mode=portfolio_cfg.reward_mode,
            inventory_penalty=portfolio_cfg.inventory_penalty,
            market_mode=portfolio_cfg.market_mode,
            return_mu=portfolio_cfg.return_mu,
            return_phi=portfolio_cfg.return_phi,
            return_sigma=portfolio_cfg.return_sigma,
            alpha_mode=portfolio_cfg.alpha_mode,
            alpha_rho=portfolio_cfg.alpha_rho,
            alpha_sigma=portfolio_cfg.alpha_sigma,
            alpha_to_return=portfolio_cfg.alpha_to_return,
            signal_noise_std=portfolio_cfg.signal_noise_std,
            cvar_n_scenarios=portfolio_cfg.cvar_n_scenarios,
            cvar_alpha=portfolio_cfg.cvar_alpha,
            price_levels_mode=portfolio_cfg.price_levels_mode,
            initial_asset_price=portfolio_cfg.initial_asset_price,
            min_asset_price=portfolio_cfg.min_asset_price,
            seed_behavior="modern" if configured_seed >= 5 else "legacy",
        )
        init_env_seed = (
            0
            if force_portfolio_zero_init
            else (effective_seed if configured_seed >= 5 else 0)
        )
    else:
        raise ValueError(
            f"Unsupported problem '{problem_name}'. Expected 'example' or 'portfolio'."
        )

    initial_state, _ = environment.reset(seed=init_env_seed)
    model.update_state(initial_state)
    if hasattr(model, "update_prev_action") and hasattr(environment, "prev_action"):
        model.update_prev_action(environment.prev_action)
    if hasattr(model, "update_cash") and hasattr(environment, "cash"):
        model.update_cash(environment.cash)
    if hasattr(model, "update_prices") and hasattr(environment, "prices"):
        model.update_prices(environment.prices)

    return model, environment


def _empirical_cvar(losses, alpha):
    losses = np.asarray(losses, dtype=float).flatten()
    if losses.size == 0:
        return 0.0
    alpha = float(np.clip(alpha, 0.0, 0.999999))
    tail_count = max(1, int(np.ceil((1.0 - alpha) * losses.size)))
    sorted_losses = np.sort(losses)
    tail_losses = sorted_losses[-tail_count:]
    return float(np.mean(tail_losses))


def sanitize_env_action(env, action):
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    action_arr = np.round(action_arr).astype(np.int32)
    if hasattr(env.action_space, "nvec"):
        nvec = env.action_space.nvec.astype(np.int32)
        action_arr = np.clip(action_arr, 0, nvec - 1)
    return action_arr


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    runtime_device = resolve_runtime_device(config.device)

    diagnostic_window = config.terminal_log_every

    m, gym_model = build_models(config, project_root)
    original_A = gym_model.A
    original_B = gym_model.B
    original_c = m.c
    original_C = m.C
    original_D = m.D
    original_E = m.E
    original_aA = m.aA.copy()
    original_aB = m.aB.copy()
    original_b = m.b.copy()
    action_size = config.model.action_size
    state = gym_model.state

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    run = wandb.init(
        name=config.name,
        mode=config.wandb_mode,
        config=config.model_dump(mode="json"),
    )

    window_size = config.plotting.window_size
    algorithm = config.training.algorithm
    solver = build_solver(config)
    training_algorithm = build_training_algorithm(
        algorithm,
        config,
        m,
        solver,
        runtime_device,
    )

    state = gym_model.state

    rollout_iters = training_algorithm.rollout_iters
    total_iters = config.total_iters

    ep_reward = 0
    economic_ep_reward = 0
    ep_rewards = []
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

    T = config.explicit_sol_time
    fathomed_counter = 0
    ep_length = 0

    comp_expected = config.comp_expected
    comp_expected_every = config.comp_expected_every

    recent_rewards = []
    recent_n_sols = []
    recent_action_mismatch = []
    recent_action_mismatch_l2 = []

    iter_counter = 0
    expected_ep_reward = None
    last_calced = 0
    for _ in tqdm(range(total_iters), desc="Total Iterations"):
        if last_calced > comp_expected_every and comp_expected:
            expected_ep_reward = calc_expected_reward(
                -original_c,
                original_A,
                original_B,
                original_C,
                original_D,
                original_E,
                T,
                state,
                solver,
            )
            last_calced = 0

        for i in tqdm(range(rollout_iters), leave=False, desc="Rollout"):
            iter_counter += 1
            ep_length += 1
            last_calced += 1
            diagnostics["step_count"] += 1

            if hasattr(m, "update_prev_action") and hasattr(gym_model, "prev_action"):
                m.update_prev_action(gym_model.prev_action)
            if hasattr(m, "update_cash") and hasattr(gym_model, "cash"):
                m.update_cash(gym_model.cash)
            if hasattr(m, "update_prices") and hasattr(gym_model, "prices"):
                m.update_prices(gym_model.prices)
            if hasattr(m, "update_scenarios") and hasattr(gym_model, "return_window"):
                m.update_scenarios(gym_model.return_window)

            decision = training_algorithm.act(state)
            action = sanitize_env_action(gym_model, decision.action)
            act_info = decision.info
            store = decision.store
            chosen_action_raw = decision.raw_action
            if not store:
                diagnostics["no_action_count"] += 1
            if act_info.get("fathomed", False):
                fathomed_counter += 1

            if store and chosen_action_raw is not None:
                executed_action = np.asarray(action, dtype=np.float32).reshape(-1)
                mismatch = float(
                    not np.allclose(chosen_action_raw, executed_action, atol=1e-6)
                )
                mismatch_l2 = float(np.linalg.norm(chosen_action_raw - executed_action))
                recent_action_mismatch.append(mismatch)
                recent_action_mismatch_l2.append(mismatch_l2)
                if len(recent_action_mismatch) > diagnostic_window:
                    recent_action_mismatch = recent_action_mismatch[-diagnostic_window:]
                if len(recent_action_mismatch_l2) > diagnostic_window:
                    recent_action_mismatch_l2 = recent_action_mismatch_l2[
                        -diagnostic_window:
                    ]

            old_state_for_buffer = np.asarray(state).copy()
            state, reward, terminated, _, info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            turnover = float(info.get("turnover", 0.0))
            risk_utilization = float(info.get("risk_utilization", 0.0))
            economic_reward = float(info.get("economic_reward", -reward))
            empirical_cvar = 0.0
            scenario_matrix = info.get("scenario_matrix", None)
            target_position = np.asarray(
                info.get("target_position", np.zeros((action_size,))), dtype=float
            )
            if (
                scenario_matrix is not None
                and hasattr(m, "cvar_mode")
                and m.cvar_mode == "on"
            ):
                scenario_matrix = np.asarray(scenario_matrix, dtype=float)
                if (
                    scenario_matrix.ndim == 2
                    and scenario_matrix.shape[1] == target_position.shape[0]
                ):
                    scenario_losses = -(scenario_matrix @ target_position)
                    empirical_cvar = _empirical_cvar(
                        scenario_losses, getattr(m, "cvar_alpha", 0.95)
                    )

            diagnostics["turnover_sum"] += turnover
            diagnostics["risk_utilization_sum"] += risk_utilization
            diagnostics["empirical_cvar_sum"] += empirical_cvar

            if store:
                training_algorithm.observe(
                    AlgorithmTransition(
                        old_state_for_buffer=np.asarray(
                            old_state_for_buffer, dtype=np.float32
                        ),
                        reward=float(reward),
                        terminated=bool(terminated),
                        action_number=action_number,
                        old_state=old_state,
                        new_state=new_state,
                        chosen_action_raw=chosen_action_raw,
                        executed_action=np.asarray(action, dtype=np.float32),
                    ),
                    act_info,
                )

            ep_reward += reward
            economic_ep_reward += economic_reward
            n_sols = 0 if (act_info is None) else act_info.get("n_sols", 0)
            diagnostics["reward_sum"] += reward
            diagnostics["economic_reward_sum"] += economic_reward
            diagnostics["reward_sq_sum"] += reward**2
            recent_rewards.append(float(reward))
            recent_n_sols.append(float(n_sols))
            if len(recent_rewards) > diagnostic_window:
                recent_rewards = recent_rewards[-diagnostic_window:]
            if len(recent_n_sols) > diagnostic_window:
                recent_n_sols = recent_n_sols[-diagnostic_window:]
            run.log(
                {
                    "reward": reward,
                    "economic_reward": economic_reward,
                    "action": action_number,
                    "n_sols": n_sols,
                    "turnover": turnover,
                    "risk_utilization": risk_utilization,
                    "empirical_cvar": empirical_cvar,
                }
            )

            if terminated or i == rollout_iters - 1:
                ep_rewards.append(ep_reward)
                diagnostics["episode_count"] += 1
                diagnostics["episode_reward_sum"] += ep_reward
                diagnostics["economic_episode_reward_sum"] += economic_ep_reward
                diagnostics["episode_length_sum"] += ep_length

                metric = {
                    "ep_reward": ep_reward,
                    "economic_ep_reward": economic_ep_reward,
                    "fathomed_counter": fathomed_counter,
                    "ep_length": ep_length,
                }

                if len(ep_rewards) == window_size:
                    metric["smooth_ep_reward"] = sum(ep_rewards) / window_size
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
                state, _ = gym_model.reset()
                if hasattr(m, "update_prev_action") and hasattr(
                    gym_model, "prev_action"
                ):
                    m.update_prev_action(gym_model.prev_action)
                if hasattr(m, "update_cash") and hasattr(gym_model, "cash"):
                    m.update_cash(gym_model.cash)
                if hasattr(m, "update_prices") and hasattr(gym_model, "prices"):
                    m.update_prices(gym_model.prices)
                if last_calced > comp_expected_every and comp_expected:
                    expected_ep_reward = calc_expected_reward(
                        -original_c,
                        original_A,
                        original_B,
                        original_C,
                        original_D,
                        original_E,
                        T,
                        state,
                        solver,
                    )
                    last_calced = 0

        update_result = training_algorithm.update(state)
        pol_grad_norm = update_result["pol_grad_norm"]
        algorithm_metrics = update_result["metrics"]

        model_params = m.get_params()
        c = model_params["c"]
        aA = model_params["aA"]
        aB = model_params["aB"]
        b = model_params["b"]

        c_change = np.sum((-original_c - c) ** 2)
        aA_change = np.sum((original_aA - aA) ** 2)
        aB_change = np.sum((original_aB - aB) ** 2)
        b_change = np.sum((original_b - b) ** 2)
        metrics = {
            "c_change": c_change,
            "aA_change": aA_change,
            "aB_change": aB_change,
            "b_change": b_change,
            "pol_grad": pol_grad_norm,
        }
        for k, v in algorithm_metrics.items():
            metrics[f"{algorithm}_{k}"] = v
        run.log(metrics)


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('cumtime').print_stats(20)
    # # cProfile.run("main()k",sort = "time")
