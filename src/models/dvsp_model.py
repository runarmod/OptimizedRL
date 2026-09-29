import numpy as np

from src.dvsp.simulator import N_ACTOR_FEATURES
from src.dvsp.solvers import PrizeCollectingLP
from src.models.model_interface import Model, VectorModelParameters


class DVSPModel(Model):
    """Prize-collecting vehicle scheduling MILP with learned prizes.

    Each postponable request v gets the prize theta_v = w . phi_v, where phi_v
    are the 14 request features of paper 02 and w are the learned parameters
    (the same generalized linear actor as paper 02's COAML pipeline). The MILP
    solved at every epoch is

        Q_w(s, y) = sum_a d_a y_a - sum_{v postponable} theta_v z_v(y),

    with z_v(y) the inflow of request v (1 if dispatched now). The prizes play
    the role of the value function V_theta in CORL eq. (5): they price the
    future cost of postponing a request.

    The parameters only enter the objective, so by the envelope theorem the
    gradient at any B&B node solution (y, z) is dQ/dw = -Phi^T z; no duals are
    needed.
    """

    def __init__(self, weights: np.ndarray):
        super().__init__()
        self.w = np.asarray(weights, dtype=float).reshape(N_ACTOR_FEATURES)
        self.features = np.zeros((0, N_ACTOR_FEATURES))
        self.lp: PrizeCollectingLP | None = None

    def update_from_environment(self, environment):
        self._update_state(environment.state)
        self.features = environment.actor_features()
        self.lp = PrizeCollectingLP.from_state(environment.epoch_state)

    def prizes(self) -> np.ndarray:
        return self.features @ self.w

    def get_LP_formulation(self):
        lp = self.lp
        n = lp.n_arcs
        return {
            "c": lp.objective(self.prizes()),
            "A_ub": lp.A_ub.toarray(),
            "b_ub": lp.b_ub.copy(),
            "A_eq": lp.A_eq.toarray(),
            "b_eq": lp.b_eq.copy(),
            "bounds": [(0, 1) for _ in range(n)],
            "integer": [1 for _ in range(n)],
            # One binary expression per postponable request: its dispatch decision.
            "branch_exprs": lp.inflow[lp.postponable].toarray(),
        }

    def lagrange_gradient(self, x_t, state, eq_duals, ineq_duals):
        served = self.lp.served(x_t)[self.lp.postponable]
        return -(self.features.T @ served)

    def get_desc_var_indices(self):
        return slice(0, self.lp.n_arcs)

    def get_params(self) -> VectorModelParameters:
        return VectorModelParameters(name="w", values=self.w.copy())

    def get_policy_params(self) -> np.ndarray:
        return self.w.astype(np.float32)

    def set_policy_params(self, theta: np.ndarray) -> None:
        self.w = np.asarray(theta, dtype=float).reshape(N_ACTOR_FEATURES)

    def update_params(self, grad, lr):
        self.w -= lr * np.asarray(grad, dtype=float)


def initial_weights(mode: str, seed: int) -> np.ndarray:
    """Initial actor weights.

    ``glorot`` matches Flux's default init of paper 02's ``Dense(14 => 1)``
    (Glorot uniform, no bias). ``zeros`` makes every prize 0, i.e. the lazy
    policy.
    """
    if mode == "zeros":
        return np.zeros(N_ACTOR_FEATURES)
    if mode == "glorot":
        limit = np.sqrt(6.0 / (N_ACTOR_FEATURES + 1))
        return np.random.default_rng(seed).uniform(-limit, limit, N_ACTOR_FEATURES)
    raise ValueError(f"Unknown DVSP init mode '{mode}'")
