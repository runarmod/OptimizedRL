"""Configuration of SRL training. Defaults are the DAP setup of the SRL codebase
(scripts/DAP/03_SRL.jl) and Table 7 of the paper."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Schedule(StrictConfig):
    """Value that moves linearly from `start` to `end` over the training episodes."""

    start: float
    end: float

    def value(self, completed_episodes: int, episodes: int) -> float:
        step = (self.start - self.end) / episodes
        value = self.start - step * completed_episodes
        return max(value, self.end) if step >= 0 else min(value, self.end)


class SrlEnvConfig(StrictConfig):
    n_items: int = Field(20, gt=0, description="Number of items.")
    assortment_size: int = Field(4, gt=0, description="Items shown per step.")
    initial_inventory: int = Field(200, gt=0, description="Initial stock of each item.")
    max_steps: int = Field(80, gt=0, description="Steps per episode.")
    n_train_seeds: int = Field(
        100,
        gt=0,
        description="Training episode e (from 1) uses seed e % n_train_seeds + 1.",
    )


class SrlValidationConfig(StrictConfig):
    episodes: int = Field(30, gt=0, description="Episodes per validation split.")
    splits: dict[str, int] = Field(
        default_factory=lambda: {"train": 1, "val": 1001},
        min_length=1,
        description="First episode seed of each split. Before every training "
        "episode, the model with the best mean over the splits is kept.",
    )


class SrlInitConfig(StrictConfig):
    reward_range: tuple[float, float] | None = Field(
        (100.0, 200.0),
        description="Accepted mean revenue of the initial actor. 00_setup.jl of the "
        "SRL codebase asks for 100-200 to avoid chance convergence. Draws outside "
        "the range are discarded. null accepts the first draw.",
    )
    episodes: int = Field(
        10, gt=0, description="Episodes from seed 1 that the initial actor is rated on."
    )
    max_draws: int = Field(10_000, gt=0, description="Give up after this many draws.")


class SrlConfig(StrictConfig):
    name: str = Field("dap srl", description="Run name.")
    seed: int = Field(14, description="Model seed: network initialization and replay sampling.")
    torch_threads: int = Field(1, gt=0, description="Threads torch may use.")

    env: SrlEnvConfig = Field(default_factory=SrlEnvConfig)
    validation: SrlValidationConfig = Field(default_factory=SrlValidationConfig)
    init: SrlInitConfig = Field(default_factory=SrlInitConfig)

    episodes: int = Field(200, gt=0, description="Training episodes.")
    iterations: int = Field(100, gt=0, description="Actor and reward-critic updates per episode.")
    batch_size: int = Field(4, gt=0, description="Transitions per update.")
    buffer_size: int = Field(8000, gt=0, description="Replay buffer capacity.")
    gamma: float = Field(0.99, description="Discount factor of the returns.")

    dynamic: bool = Field(
        True,
        description="Use the future critic for the target action (the paper's "
        "dynamic setting). False uses the reward critic only.",
    )
    future_critic_passes: int = Field(
        50,
        ge=0,
        description="Shuffled passes of the future critic over each new episode "
        "(round(iterations / 2) in the SRL codebase).",
    )

    target_samples: int = Field(40, gt=0, description="Perturbed candidates for the target action.")
    # The SRL codebase passes these as `MvNormal(θ, σ * I)`, whose covariance is
    # σ I, so the noise has standard deviation sqrt(σ).
    sigma_f: Schedule = Field(
        default_factory=lambda: Schedule(start=2.0, end=1.0),
        description="Exploration noise variance during experience collection.",
    )
    sigma_b: Schedule = Field(
        default_factory=lambda: Schedule(start=2.0, end=1.0),
        description="Perturbation noise variance of the target candidates.",
    )
    temperature: Schedule = Field(
        default_factory=lambda: Schedule(start=1.0, end=1.0),
        description="Softmax temperature of the target action.",
    )
    lr: Schedule = Field(
        default_factory=lambda: Schedule(start=1e-3, end=5e-4),
        description="Adam learning rate of the actor and both critics.",
    )
    grad_clip: float = Field(1e-3, gt=0, description="Elementwise gradient clip before Adam.")

    fy_epsilon: float = Field(1.0, gt=0, description="Perturbation scale of the Fenchel-Young loss.")
    fy_samples: int = Field(20, gt=0, description="Perturbations of the Fenchel-Young loss.")


def load_srl_config(path: Path) -> SrlConfig:
    with path.open() as config_file:
        raw = yaml.safe_load(config_file) or {}
    return SrlConfig.model_validate(raw)
