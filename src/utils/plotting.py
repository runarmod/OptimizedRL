import os

import wandb
from src.config.config_models import AppConfig
from src.models.model_interface import ModelParameters


class Plotter:
    def __init__(self, config: AppConfig):

        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        self.run = wandb.init(
            name=config.name,
            mode=config.wandb_mode,
            config=config.model_dump(mode="json"),
        )
        self.window_size = config.plotting.window_size
        self.episode_rewards = []
        self.episode_reward = 0.0
        self.episode_environment_metrics = {}
        self.fathomed_counter = 0
        self.episode_length = 0

    def log_step(
        self,
        reward: float,
        action: int,
        n_sols: int,
        environment_metrics: dict[str, float],
        fathomed: bool,
        episode_done: bool,
        expected_ep_reward: float | None = None,
    ) -> None:
        step_metrics = {
            "reward": reward,
            "action": action,
            "n_sols": n_sols,
            **environment_metrics,
        }
        self.run.log(step_metrics)

        self.episode_reward += reward
        self.episode_length += 1
        if fathomed:
            self.fathomed_counter += 1
        for name, value in environment_metrics.items():
            self.episode_environment_metrics[name] = (
                self.episode_environment_metrics.get(name, 0.0) + value
            )

        if not episode_done:
            return

        self.episode_rewards.append(self.episode_reward)
        episode_metrics = {
            "ep_reward": self.episode_reward,
            "fathomed_counter": self.fathomed_counter,
            "ep_length": self.episode_length,
        }
        if "economic_reward" in self.episode_environment_metrics:
            episode_metrics["economic_ep_reward"] = self.episode_environment_metrics[
                "economic_reward"
            ]

        if len(self.episode_rewards) == self.window_size:
            episode_metrics["smooth_ep_reward"] = (
                sum(self.episode_rewards) / self.window_size
            )
            self.episode_rewards = []
        if expected_ep_reward is not None:
            episode_metrics["expected_ep_reward"] = expected_ep_reward
            episode_metrics["distance_from_opt_pol"] = (
                expected_ep_reward - self.episode_reward
            )
        self.run.log(episode_metrics)
        self.episode_reward = 0.0
        self.episode_environment_metrics = {}
        self.fathomed_counter = 0
        self.episode_length = 0

    def log_update(
        self,
        algorithm: str,
        update_result: dict,
        original_params: ModelParameters,
        model_params: ModelParameters,
    ):
        metrics = {
            "c_change": ((-original_params.c - model_params.c) ** 2).sum(),
            "aA_change": ((original_params.aA - model_params.aA) ** 2).sum(),
            "aB_change": ((original_params.aB - model_params.aB) ** 2).sum(),
            "b_change": ((original_params.b - model_params.b) ** 2).sum(),
            "pol_grad": update_result["pol_grad_norm"],
        }
        for name, value in update_result["metrics"].items():
            metrics[f"{algorithm}_{name}"] = value
        self.run.log(metrics)
