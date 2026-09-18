import numpy as np

from src.config.config_models import AppConfig
from src.models.model_interface import Model
from src.solvers.solver_interface import Solver
from src.training_algorithm.training_algorithm_interface import (
    AlgorithmDecision,
    AlgorithmTransition,
    TrainingAlgorithm,
)
from src.training_algorithm.vanilla_gradient import actor, gae


class VanillaGradientAlgorithm(TrainingAlgorithm):
    def __init__(
        self, config: AppConfig, model: Model, solver: Solver, runtime_device: str
    ):
        critic = gae.GAE(
            [config.model.state_size, 128, 128, 1],
            config.critic.lr,
            config.critic.df,
            config.critic.eps,
            0.1,
            runtime_device,
        )
        self.agent = actor.Actor(
            model,
            solver,
            critic,
            beta=config.actor.beta,
            lr=config.actor.lr,
            df=config.critic.df,
            nn_sample=config.actor.nn_sample,
            sampled_grad=config.actor.sampled_grad,
        )
        self.rollout_iters = config.rollout_iters
        self.training_iters = config.train_iters
        self.sample = config.actor.sample
        self.num_samples = config.actor.num_samples
        self.action_size = config.model.action_size

    def act(self, state: np.ndarray) -> AlgorithmDecision:
        action, info = self.agent.act(state)
        return AlgorithmDecision(action=action, info=info, store=True)

    def observe(self, transition: AlgorithmTransition, info: dict) -> None:
        self.agent.update_buffers(
            transition.reward,
            transition.action_number,
            transition.old_state,
            transition.new_state,
            info["nab"],
            info["t_nab"],
        )

    def update(self, state: np.ndarray) -> dict:
        _ = state
        pol_grad = self.agent.train(
            iters=self.training_iters,
            sample=self.sample,
            num_samples=self.num_samples,
        )
        return {
            "pol_grad_norm": float(np.linalg.norm(pol_grad)),
            "metrics": {},
        }
