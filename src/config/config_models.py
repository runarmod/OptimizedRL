from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScipConfig(StrictConfig):
    verbose: bool = Field(description="Show SCIP solver output.")


class ScipBruteConfig(StrictConfig):
    verbose: bool = Field(description="Show SCIP solver output.")
    disable_heuristics: bool = Field(description="Disable SCIP primal heuristics.")
    disable_presolve: bool = Field(description="Disable SCIP presolving.")
    disable_separating: bool = Field(
        description="Disable SCIP cutting-plane separation."
    )
    disable_propagation: bool = Field(
        description="Disable SCIP constraint propagation."
    )
    disable_conflict_analysis: bool = Field(
        description="Disable SCIP conflict analysis."
    )
    disable_symmetry: bool = Field(description="Disable SCIP symmetry handling.")
    prefer_most_fractional_branching: bool = Field(
        description="Prefer the custom most-fractional branching rule."
    )
    prefer_breadth_first: bool = Field(
        description="Prefer breadth-first node selection."
    )
    tighten_integer_projected_bounds: bool = Field(
        description="Project integer candidates onto fixed integer bounds."
    )
    mimic_bnb_pool_filter: bool = Field(
        description="Filter the candidate pool using BnB-like ordering."
    )
    prefer_depth_first: bool = Field(description="Prefer depth-first node selection.")


class TrainingConfig(StrictConfig):
    algorithm: Literal["vanilla_gradient", "ppo"] = Field(
        description="Training algorithm."
    )
    solver: Literal["scip", "scip_brute", "bnb"] = Field(
        description="MILP solver implementation."
    )


class ModelConfig(StrictConfig):
    state_size: int = Field(gt=0, description="Number of state variables.")
    action_size: int = Field(gt=0, description="Number of action variables.")
    n_cons: int = Field(gt=0, description="Number of constraints.")
    n_value_func: int = Field(gt=0, description="Number of value-function pieces.")
    penalty_factor: float = Field(
        ge=0, description="Penalty factor used in the optimization model."
    )


class GymConfig(StrictConfig):
    pf: int = Field(gt=0, description="Environment planning or penalty factor.")
    noise_std: float = Field(
        ge=0, description="Standard deviation of environment transition noise."
    )


class PlottingConfig(StrictConfig):
    window_size: int = Field(
        gt=0, description="Number of episodes used for smoothed metrics."
    )


class CriticConfig(StrictConfig):
    lr: float = Field(gt=0, description="Vanilla-gradient critic learning rate.")
    df: float = Field(ge=0, le=1, description="Vanilla-gradient discount factor.")
    eps: float = Field(
        gt=0, description="Vanilla-gradient critic convergence tolerance."
    )


class ActorConfig(StrictConfig):
    sampled_grad: bool = Field(
        description="Use a sampled LP gradient in vanilla-gradient training."
    )
    nn_sample: bool = Field(
        description="Use neighborhood sampling for fractional solutions."
    )
    lr: float = Field(gt=0, description="Vanilla-gradient actor learning rate.")
    beta: float = Field(
        gt=0, description="Vanilla-gradient policy inverse temperature."
    )
    sample: bool = Field(description="Subsample vanilla-gradient training data.")
    num_samples: float = Field(
        gt=0, description="Fraction of buffered samples to use when sampling."
    )


class PortfolioModelConfig(StrictConfig):
    lr_mult_aA: float = Field(
        gt=0, description="Learning-rate multiplier for matrix aA."
    )
    lr_mult_aB: float = Field(
        gt=0, description="Learning-rate multiplier for matrix aB."
    )
    lr_mult_b: float = Field(gt=0, description="Learning-rate multiplier for vector b.")
    position_dynamics_mode: Literal["legacy", "trade_stateful"] = Field(
        description="Position dynamics formulation."
    )


class PortfolioEnvConfig(StrictConfig):
    transaction_cost: float = Field(ge=0, description="Per-unit transaction cost.")
    holding_cost: float = Field(ge=0, description="Per-unit holding cost.")
    budget_cap: float | None = Field(description="Optional portfolio budget cap.")
    initial_cash: float = Field(ge=0, description="Initial cash balance.")
    cash_interest_rate: float = Field(description="Cash interest rate per step.")
    risk_cap: float | None = Field(description="Optional portfolio risk cap.")
    risk_weight: float = Field(ge=0, description="Portfolio risk constraint weight.")
    asset_max_position: float | None = Field(
        description="Optional per-asset position cap."
    )
    action_mode: Literal["absolute", "target"] = Field(
        description="Meaning of environment actions."
    )
    reward_mode: Literal["economic", "legacy"] = Field(
        description="Environment reward calculation mode."
    )
    inventory_penalty: float = Field(ge=0, description="Penalty applied to inventory.")
    market_mode: Literal["linear", "returns"] = Field(
        description="Market dynamics mode."
    )
    return_mu: float = Field(description="Mean return in returns market mode.")
    return_phi: float = Field(description="Return autoregression coefficient.")
    return_sigma: float = Field(ge=0, description="Return noise standard deviation.")
    alpha_mode: Literal["off", "on"] = Field(
        description="Enable latent alpha dynamics."
    )
    alpha_rho: float = Field(description="Latent alpha autoregression coefficient.")
    alpha_sigma: float = Field(
        ge=0, description="Latent alpha noise standard deviation."
    )
    alpha_to_return: float = Field(description="Scale from latent alpha to returns.")
    signal_noise_std: float = Field(
        ge=0, description="Observation signal noise standard deviation."
    )
    return_signal_scale: float = Field(
        description="Scale applied to return signals in the model objective."
    )
    cvar_mode: Literal["off", "on"] = Field(
        description="Enable CVaR constraints in the model."
    )
    cvar_cap: float = Field(description="CVaR cap.")
    cvar_alpha: float = Field(gt=0, lt=1, description="CVaR tail confidence level.")
    cvar_n_scenarios: int = Field(gt=0, description="Number of CVaR scenarios.")
    cvar_obj_weight: float = Field(ge=0, description="CVaR objective weight.")
    price_levels_mode: Literal["off", "on"] = Field(
        description="Use current asset prices in the model."
    )
    initial_asset_price: float = Field(gt=0, description="Initial asset price.")
    min_asset_price: float = Field(gt=0, description="Minimum asset price.")


class PpoConfig(StrictConfig):
    nn_sample: bool | None = Field(
        None, description="Optional PPO override for fractional-solution sampling."
    )
    minimize_env_reward: bool = Field(
        description="Negate environment rewards for PPO optimization."
    )
    normalize_rewards: bool = Field(
        description="Normalize rewards before advantage calculation."
    )
    reward_norm_eps: float = Field(
        gt=0, description="Epsilon used for reward normalization."
    )
    reward_clip: float | None = Field(
        description="Optional absolute reward clipping limit."
    )
    gamma: float = Field(ge=0, le=1, description="PPO discount factor.")
    gae_lambda: float = Field(ge=0, le=1, description="GAE trace-decay factor.")
    clip_param: float = Field(
        ge=0, description="PPO probability-ratio clipping parameter."
    )
    entropy_coef: float = Field(ge=0, description="PPO entropy loss coefficient.")
    value_coef: float = Field(ge=0, description="PPO value loss coefficient.")
    actor_lr: float = Field(gt=0, description="PPO policy learning rate.")
    critic_lr: float = Field(gt=0, description="PPO value-network learning rate.")
    policy_beta: float = Field(gt=0, description="PPO policy inverse temperature.")
    normalize_obj_values: bool = Field(
        description="Normalize MILP objective values before policy logits."
    )
    obj_norm_eps: float = Field(
        gt=0, description="Epsilon used for objective normalization."
    )
    opt_epochs: int = Field(gt=0, description="PPO optimization epochs per rollout.")
    mini_batch_size: int = Field(gt=0, description="PPO mini-batch size.")
    rollout_iters: int = Field(
        gt=0, description="PPO rollout steps per training iteration."
    )
    target_kl: float = Field(ge=0, description="Optional PPO target KL threshold.")
    normalize_adv: bool = Field(
        description="Normalize advantages within PPO mini-batches."
    )


class AppConfig(StrictConfig):
    name: str = Field(description="Experiment name.")
    wandb_mode: str = Field(description="Weights & Biases execution mode.")
    problem: Literal["example", "portfolio"] = Field(
        description="Optimization problem implementation."
    )
    portfolio_zero_init_seeds: list[int] = Field(
        description="Portfolio seeds that use deterministic zero initialization."
    )
    numpy_seed: int = Field(description="Random seed for NumPy and environment setup.")
    load: bool = Field(description="Load model parameters from load_path.")
    load_path: Path = Field(description="Path to saved model parameters.")
    device: str = Field(description="Requested Torch device, such as cpu or cuda.")
    scip: ScipConfig
    scip_brute: ScipBruteConfig
    training: TrainingConfig
    model: ModelConfig
    gym: GymConfig
    plotting: PlottingConfig
    critic: CriticConfig
    actor: ActorConfig
    portfolio_model: PortfolioModelConfig
    portfolio_env: PortfolioEnvConfig
    ppo: PpoConfig
    train_iters: int = Field(
        gt=0, description="Vanilla-gradient training epochs per rollout."
    )
    rollout_iters: int = Field(
        gt=0, description="Vanilla-gradient rollout steps per training iteration."
    )
    total_iters: int = Field(gt=0, description="Total outer training iterations.")
    explicit_sol_time: int = Field(
        gt=0, description="Time horizon used by expected-reward evaluation."
    )
    comp_expected: bool = Field(
        description="Compare against an expected reward when enabled."
    )
    comp_expected_every: int = Field(
        gt=0, description="Iterations between expected-reward calculations."
    )
    terminal_log_every: int = Field(
        gt=0, description="Window size for rolling terminal diagnostics."
    )
