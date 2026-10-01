import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.config.config_loader import load_config
from src.config.config_models import AppConfig
from src.dap.evaluation import DAPValidation, find_initial_theta
from src.gym_envs import dap_env, example_env, portfolio_env
from src.gym_envs.env_interface import Env
from src.models import dap_model, example_model, portfolio_model
from src.models.model_interface import Model
from src.solvers import bnb, enumeration, exact_assortment, scip, scip_brute
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
from src.utils.plotting import Plotter


def build_training_algorithm(
    algorithm: str,
    config: AppConfig,
    model: Model,
    solver: Solver,
    runtime_device: str,
    state_dim: int,
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
                state_dim=state_dim,
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
        f"training.algorithm must be 'vanilla_gradient' or 'ppo' Got '{algorithm}'."
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

        case "enumeration":
            return enumeration.BranchEnumerationSolver(
                max_pool_size=config.enumeration.max_pool_size,
                max_depth=config.enumeration.max_depth,
                integral_only=config.enumeration.integral_only,
            )

        case "exact_assortment":
            return exact_assortment.ExactAssortmentSolver(
                max_candidates=config.exact_assortment.max_candidates,
            )

    raise ValueError(
        "training.solver must be one of 'scip', 'scip_brute', 'bnb', "
        f"'enumeration' or 'exact_assortment'. Got '{config.training.solver}'."
    )


def resolve_runtime_device(configured_device):
    device = str(configured_device).strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def build_dap(
    config: AppConfig, project_root: Path
) -> tuple[Model, Env, DAPValidation | None]:
    cfg = config.dap
    if config.load:
        if config.load_path is None:
            raise ValueError("Config 'load' is True but 'load_path' is not set.")
        with resolve_path(config.load_path, project_root).open() as params_file:
            theta = np.array(yaml.safe_load(params_file)["theta"], dtype=float)
    else:
        theta, tries, reward = find_initial_theta(
            cfg.n_value_pieces, config.numpy_seed, cfg.init_reward_range
        )
        print(f"Initial parameters: {tries} draw(s), deterministic reward {reward:.1f}")

    model = dap_model.DAPModel(
        theta,
        cfg.n_value_pieces,
        k=cfg.k,
        nns_neighbours=cfg.nns_neighbours,
        seed=config.numpy_seed,
    )
    first, last = cfg.train_seeds
    environment = dap_env.DAPEnv(
        range(first, last + 1), seed_order=cfg.seed_order, seed=config.numpy_seed
    )
    environment.reset(seed=config.numpy_seed)
    model.update_from_environment(environment)

    validation = None
    if cfg.eval_every > 0:
        validation = DAPValidation(
            cfg.n_value_pieces, n_episodes=cfg.eval_episodes, selection=cfg.selection
        )
    return model, environment, validation


def save_best_params(
    config: AppConfig, project_root: Path, theta: np.ndarray, info: dict
) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.name)
    path = project_root / "params" / f"{config.problem}_{safe_name}_best.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as params_file:
        yaml.safe_dump({"theta": np.asarray(theta).tolist(), **info}, params_file)
    return path


def build_model_and_env(config: AppConfig, project_root: Path) -> tuple[Model, Env]:
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
            cvar_mode=portfolio_cfg.cvar_mode,
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

    environment.reset(seed=init_env_seed)
    model.update_from_environment(environment)

    return model, environment


def sanitize_env_action(env, action):
    if hasattr(env, "sanitize_action"):
        return env.sanitize_action(action)
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    action_arr = np.round(action_arr).astype(np.int32)
    nvec = env.action_space.nvec.astype(np.int32)
    action_arr = np.clip(action_arr, 0, nvec - 1)
    return action_arr


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CORL MILP policy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config.yaml",
        help="Path to the YAML config (default: config.yaml).",
    )
    return parser.parse_args()


def main():
    project_root = Path(__file__).resolve().parent
    args = parse_args(project_root)
    config = load_config(resolve_path(args.config, Path.cwd()))
    runtime_device = resolve_runtime_device(config.device)

    validation = None
    eval_every = 0
    if config.problem == "dap":
        model, env, validation = build_dap(config, project_root)
        eval_every = config.dap.eval_every
    else:
        model, env = build_model_and_env(config, project_root)
    if config.comp_expected:
        original_A = env.A
        original_B = env.B
        original_C = model.C
        original_D = model.D
        original_E = model.E
    original_model_params = model.get_params()
    state = env.state

    plotter = Plotter(config)
    algorithm = config.training.algorithm
    solver = build_solver(config)
    training_algorithm = build_training_algorithm(
        algorithm,
        config,
        model,
        solver,
        runtime_device,
        state_dim=int(np.asarray(env.state).reshape(-1).size),
    )

    state = env.state

    rollout_iters = training_algorithm.rollout_iters
    total_iters = config.total_iters

    T = config.explicit_sol_time

    comp_expected = config.comp_expected
    comp_expected_every = config.comp_expected_every

    expected_ep_reward = None
    last_calced = 0
    for iteration in tqdm(range(total_iters), desc="Total Iterations"):
        if validation is not None and iteration % eval_every == 0:
            eval_metrics = validation.evaluate(model.get_policy_params())
            eval_metrics[f"{config.problem}/iteration"] = iteration
            plotter.log_evaluation(eval_metrics)
            problem_config = getattr(config, config.problem, None)
            save_best = getattr(problem_config, "save_best", False)
            if validation.is_new_best(eval_metrics) and save_best:
                save_best_params(
                    config,
                    project_root,
                    model.get_policy_params(),
                    {"iteration": iteration, **eval_metrics},
                )

        if last_calced > comp_expected_every and comp_expected:
            expected_ep_reward = calc_expected_reward(
                -original_model_params.c,
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
            last_calced += 1
            model.update_from_environment(env)

            decision = training_algorithm.act(state)
            action = sanitize_env_action(env, decision.action)
            act_info = decision.info
            store = decision.store
            chosen_action_raw = decision.raw_action
            old_state_for_buffer = np.asarray(state).copy()
            state, reward, terminated, _, info = env.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            n_sols = 0 if act_info is None else act_info.get("n_sols", 0)
            environment_metrics = env.get_plot_metrics(info)

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

            episode_done = terminated or (
                config.reset_env_each_rollout and i == rollout_iters - 1
            )
            plotter.log_step(
                reward=float(reward),
                action=action_number,
                n_sols=n_sols,
                environment_metrics=environment_metrics,
                fathomed=act_info.get("fathomed", False),
                episode_done=episode_done,
                expected_ep_reward=(expected_ep_reward if comp_expected else None),
            )
            if episode_done:
                if comp_expected and expected_ep_reward is not None:
                    expected_ep_reward = None
                state, _ = env.reset()
                model.update_from_environment(env)
                if last_calced > comp_expected_every and comp_expected:
                    expected_ep_reward = calc_expected_reward(
                        -original_model_params.c,
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
        model_params = model.get_params()
        plotter.log_update(
            algorithm, update_result, original_model_params, model_params
        )


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('cumtime').print_stats(20)
    # # cProfile.run("main()k",sort = "time")
