from collections.abc import Sequence

import numpy as np

from src.dvsp.instance import VSPInstance
from src.dvsp.simulator import DynamicVSP, EpochState, routes_from_arc_values
from src.gym_envs.env_interface import Env


class DVSPEnv(Env):
    """Training environment over a set of DVSP instances.

    Every reset draws one instance. With ``scenario_mode="fixed"`` each
    instance always replays the same request stream (paper 02's
    ``is_deterministic=true``); with ``"random"`` a new stream is drawn.

    Actions are arc vectors of the current epoch graph (the MILP's decision
    variables); the reward is minus the total duration of the dispatched routes.
    ``self.state`` is a fixed-size summary for the value network, while the
    model reads the variable-size epoch problem from ``epoch_state`` and
    ``actor_features()``.
    """

    def __init__(
        self,
        instances: Sequence[VSPInstance],
        max_requests_per_epoch: int = 10,
        scenario_mode: str = "fixed",
        scenario_seed: int = 0,
        serve_final_epoch: bool = False,
        seed: int = 0,
    ):
        if not instances:
            raise ValueError("DVSPEnv needs at least one instance")
        if scenario_mode not in ("fixed", "random"):
            raise ValueError("scenario_mode must be 'fixed' or 'random'")
        self.sims = [
            DynamicVSP(inst, max_requests_per_epoch, scenario_seed)
            for inst in instances
        ]
        self.scenario_mode = scenario_mode
        self.scenario_seed = scenario_seed
        self.serve_final_epoch = serve_final_epoch
        self.rng = np.random.default_rng(seed)
        self.sim = self.sims[0]
        self.state = None

    @property
    def epoch_state(self) -> EpochState:
        return self.sim.state

    def actor_features(self) -> np.ndarray:
        return self.sim.actor_features()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(100):
            self.sim = self.sims[int(self.rng.integers(len(self.sims)))]
            scenario = (
                self.scenario_seed
                if self.scenario_mode == "fixed"
                else int(self.rng.integers(2**31 - 1))
            )
            self.sim.reset(scenario)
            self.sim.next_epoch()
            # Without the final epoch, an instance with a single epoch has no decisions.
            if self.serve_final_epoch or not self.sim.is_terminated():
                break
        else:
            raise RuntimeError("No DVSP instance has a decision epoch")
        self.state = self.sim.state_summary()
        return self.state, {"instance": self.sim.instance.name}

    def sanitize_action(self, action):
        return np.rint(np.clip(np.asarray(action, dtype=float).reshape(-1), 0, 1))

    def step(self, action):
        state = self.sim.state
        y = self.sanitize_action(action)
        if y.shape[0] != len(state.arcs):
            raise ValueError(
                f"action has {y.shape[0]} arcs, epoch graph has {len(state.arcs)}"
            )
        routes = routes_from_arc_values(state.arcs, y)
        if not state.is_feasible(routes):
            raise ValueError(
                f"infeasible DVSP action in epoch {self.sim.current_epoch}: "
                "the solver must return integral, feasible candidates"
            )

        old_state = self.state
        dispatched = np.zeros(state.n_locations, dtype=bool)
        for route in routes:
            dispatched[route] = True
        cost = self.sim.apply_routes(routes)
        epoch = self.sim.current_epoch

        if self.serve_final_epoch and self.sim.is_terminated():
            terminated = True
        else:
            self.sim.next_epoch()
            terminated = (not self.serve_final_epoch) and self.sim.is_terminated()
        self.state = self.sim.state_summary()

        info = {
            "action": int(dispatched.sum()),
            "old_state": old_state,
            "new_state": self.state,
            "route_cost": cost,
            "n_routes": len(routes),
            "n_dispatched": int(dispatched.sum()),
            "n_postponed": int((state.is_postponable & ~dispatched).sum()),
            "n_must_dispatch": state.n_must_dispatch,
            "epoch": epoch,
            "instance": self.sim.instance.name,
        }
        return self.state, -cost, terminated, False, info

    def get_plot_metrics(self, info: dict) -> dict[str, float]:
        return {
            "route_cost": float(info["route_cost"]),
            "n_routes": float(info["n_routes"]),
            "n_dispatched": float(info["n_dispatched"]),
            "n_postponed": float(info["n_postponed"]),
        }
