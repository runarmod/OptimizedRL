from itertools import combinations

import numpy as np

from src.gym_envs.dap_env import N_ITEM_FEATURES
from src.models.model_interface import Model, ParameterSnapshot, VectorModelParameters

# Item features phi_{t,i}: the environment's per-item observation plus a constant.
N_FEATURES = N_ITEM_FEATURES + 1

# Keeps the pair features phi_{t,ij} on the scale of the item features.
PAIR_FEATURE_SCALE = 10.0

# Nearest-neighbour completion of fractional candidates: sample among this many
# nearest assortments with weights exp(-beta * L1 distance), as knn_branch_sample.
NN_SAMPLE_SIZE = 9
NN_SAMPLE_BETA = 0.5

INTEGRAL_TOL = 1e-6


def _var_index(var_id: int | str) -> int | None:
    """Index of a branched variable named by its index or a name like "x_3"."""
    if isinstance(var_id, int):
        return var_id
    suffix = str(var_id).rpartition("x_")[2]
    return int(suffix) if suffix.isdigit() else None


class DapModel(Model):
    """MILP policy for the dynamic assortment problem.

    The learned parameters theta = (a, w, (psi_l, c_l, b_l) for each value piece l)
    are stored as one flat vector. They define the coefficients

        rho_i = a^T phi_i, q_ij = w^T phi_ij, g_li = psi_l^T phi_i,
        d_l = b_l + c_l^T phi_bar,

    of the MILP

        min  -sum_i rho_i x_i + sum_{i<j} q_ij y_ij + v
        s.t. sum_i x_i = K,
             v >= d_l + sum_i g_li x_i                 for every value piece l,
             y_ij >= x_i + x_j - 1, y_ij <= x_i, y_ij <= x_j   for every pair i<j,
             x binary, 0 <= y <= 1, v free.

    Variables are ordered [x (n_items), y (one per pair i<j), v].
    """

    POLICY_PARAM_ORDER = ("theta",)

    def __init__(
        self,
        theta: np.ndarray,
        n_items: int,
        assortment_size: int,
        n_value_pieces: int,
    ):
        super().__init__()
        self.n_items = n_items
        self.assortment_size = assortment_size
        self.n_value_pieces = n_value_pieces
        self.n_desc_vars = n_items

        expected_size = 2 * N_FEATURES + n_value_pieces * (2 * N_FEATURES + 1)
        if theta.shape != (expected_size,):
            raise ValueError(
                f"theta must have shape ({expected_size},), got {theta.shape}"
            )
        self.theta = theta.astype(float)

        self.pair_i, self.pair_j = np.triu_indices(n_items, k=1)
        self.n_pairs = len(self.pair_i)

        self.phi = None
        self.phi_bar = None
        self.phi_pairs = None

    def split_theta(self) -> tuple[np.ndarray, ...]:
        """Views (a, w, psi, c, b) of theta, with psi and c of shape
        (n_value_pieces, F) and b of shape (n_value_pieces,)."""
        a = self.theta[:N_FEATURES]
        w = self.theta[N_FEATURES : 2 * N_FEATURES]
        pieces = self.theta[2 * N_FEATURES :].reshape(
            self.n_value_pieces, 2 * N_FEATURES + 1
        )
        psi = pieces[:, :N_FEATURES]
        c = pieces[:, N_FEATURES : 2 * N_FEATURES]
        b = pieces[:, 2 * N_FEATURES]
        return a, w, psi, c, b

    def get_desc_var_indices(self):
        return slice(self.n_items)

    def get_params(self) -> ParameterSnapshot:
        return VectorModelParameters(name="theta", values=self.theta.copy())

    def _features(self, state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Item features phi (n_items, F), mean item phi_bar (F,) and pair
        features phi_pairs (n_pairs, F)."""
        item_obs = np.asarray(state, dtype=float).reshape(self.n_items, N_ITEM_FEATURES)
        phi = np.hstack((item_obs, np.ones((self.n_items, 1))))
        phi_bar = phi.mean(axis=0)
        phi_pairs = phi[self.pair_i] * phi[self.pair_j] / PAIR_FEATURE_SCALE
        return phi, phi_bar, phi_pairs

    def update_from_environment(self, environment):
        self._update_state(environment.state)
        self.phi, self.phi_bar, self.phi_pairs = self._features(self.s_t)

    def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Immediate revenue rho (n_items,), pair cost q (n_pairs,), value slopes
        g (n_value_pieces, n_items) and value constants d (n_value_pieces,)."""
        a, w, psi, c, b = self.split_theta()
        rho = self.phi @ a
        q = self.phi_pairs @ w
        g = psi @ self.phi.T
        d = b + c @ self.phi_bar
        return rho, q, g, d

    def _assortment_solution(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Optimal (x, y, v) of the MILP with the binary assortment x fixed, and the
        index of the active value piece."""
        _, _, g, d = self.coefficients()
        y = x[self.pair_i] * x[self.pair_j]
        pieces = d + g @ x
        active = int(np.argmax(pieces))
        return np.hstack((x, y, pieces[active])), active

    def assortment_value(self, x: np.ndarray) -> float:
        """Objective value Q_theta(s_t, x) of a binary assortment x."""
        rho, q, _, _ = self.coefficients()
        solution, _ = self._assortment_solution(np.asarray(x, dtype=float))
        return float(np.hstack((-rho, q, 1.0)) @ solution)

    def _branching_bounds(self, sol: dict) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of the assortment variables at a candidate's node."""
        x_bounds = sol["bounds"][: self.n_items]
        lower = np.array([lb for lb, _ in x_bounds], dtype=float)
        upper = np.array([ub for _, ub in x_bounds], dtype=float)
        for var_id, op, value in sol.get("conds", []):
            idx = _var_index(var_id)
            if idx is None or not 0 <= idx < self.n_items:
                continue
            if op == ">=":
                lower[idx] = max(lower[idx], value)
            else:
                upper[idx] = min(upper[idx], value)
        return lower, upper

    def complete_assortment(self, sol: dict) -> np.ndarray:
        """Assortment a candidate is executed as.

        An integral candidate is used as it is. A fractional one is completed by
        sampling among the assortments nearest to its x (in L1 distance) that are
        consistent with its node's branching decisions.
        """
        x = np.asarray(sol["x"][: self.n_items], dtype=float)
        x_round = np.round(x)
        if (
            np.allclose(x, x_round, atol=INTEGRAL_TOL)
            and x_round.sum() == self.assortment_size
        ):
            return x_round

        lower, upper = self._branching_bounds(sol)
        shown = lower > 0.5
        free = np.flatnonzero((upper > 0.5) & ~shown)
        n_missing = self.assortment_size - int(shown.sum())
        if not 0 <= n_missing <= len(free):
            raise ValueError(
                f"No assortment of {self.assortment_size} items is consistent with "
                f"the candidate's branching ({int(shown.sum())} items fixed shown, "
                f"{len(free)} free)."
            )

        x_hat = shown.astype(float)
        if n_missing == 0:
            return x_hat

        options = np.array(list(combinations(free, n_missing)))
        # L1 distance to x up to a constant shared by all options: showing free
        # item i costs 1 - x_i instead of x_i.
        dist = (1.0 - 2.0 * x[options]).sum(axis=1)
        nearest = np.argsort(dist, kind="stable")[:NN_SAMPLE_SIZE]
        weights = np.exp(-NN_SAMPLE_BETA * (dist[nearest] - dist[nearest].min()))
        choice = np.random.choice(len(nearest), p=weights / weights.sum())
        x_hat[options[nearest[choice]]] = 1.0
        return x_hat

    def complete_candidates(self, sol_pool: list[dict]) -> list[dict]:
        """Replace every candidate by the assortment it is executed as.

        Each candidate becomes the LP solution of its node with x fixed to its
        completed assortment x_hat, so it is scored by the exact value
        Q_theta(s_t, x_hat) and its duals give the exact gradient of that value.
        """
        return [self._completed_candidate(sol) for sol in sol_pool]

    def _completed_candidate(self, sol: dict) -> dict:
        node = sol["node"]
        x_hat = self.complete_assortment(sol)
        solution, active = self._assortment_solution(x_hat)

        # Only the active value piece binds, with dual 1 (marginal -1). x and y lie
        # on a bound, so their bound multipliers take up the reduced costs; v has
        # a zero reduced cost since the value-piece duals sum to 1.
        ineqlin = np.zeros(node["A_ub"].shape[0])
        ineqlin[active] = -1.0
        eqlin = np.zeros(node["A_eq"].shape[0])
        reduced_costs = node["c"] - ineqlin @ node["A_ub"]
        n_binary = self.n_items + self.n_pairs
        lower = np.zeros_like(reduced_costs)
        upper = np.zeros_like(reduced_costs)
        at_one = solution[:n_binary] == 1.0
        upper[:n_binary] = np.where(at_one, reduced_costs[:n_binary], 0.0)
        lower[:n_binary] = np.where(at_one, 0.0, reduced_costs[:n_binary])

        bounds = list(node["bounds"])
        bounds[: self.n_items] = [(value, value) for value in x_hat]

        return {
            **sol,
            "fun": float(node["c"] @ solution),
            "x": solution,
            "eqlin": eqlin,
            "ineqlin": ineqlin,
            "lower": lower,
            "upper": upper,
            "fathomed": False,
            "bounds": bounds,
            "status": "completed",
            "lp_fun": float(sol["fun"]),
        }

    def get_LP_formulation(self):
        rho, q, g, d = self.coefficients()
        n, n_pairs, n_pieces = self.n_items, self.n_pairs, self.n_value_pieces
        n_vars = n + n_pairs + 1
        y_cols = n + np.arange(n_pairs)
        v_col = n + n_pairs

        c = np.hstack((-rho, q, 1.0))

        # Value pieces first, so the first n_value_pieces inequality duals belong
        # to them: sum_i g_li x_i - v <= -d_l.
        A_value = np.zeros((n_pieces, n_vars))
        A_value[:, :n] = g
        A_value[:, v_col] = -1.0
        b_value = -d

        # McCormick constraints forcing y_ij = x_i x_j for binary x.
        rows = np.arange(n_pairs)
        A_lower = np.zeros((n_pairs, n_vars))  # x_i + x_j - y_ij <= 1
        A_lower[rows, self.pair_i] = 1.0
        A_lower[rows, self.pair_j] = 1.0
        A_lower[rows, y_cols] = -1.0
        A_upper_i = np.zeros((n_pairs, n_vars))  # y_ij - x_i <= 0
        A_upper_i[rows, y_cols] = 1.0
        A_upper_i[rows, self.pair_i] = -1.0
        A_upper_j = np.zeros((n_pairs, n_vars))  # y_ij - x_j <= 0
        A_upper_j[rows, y_cols] = 1.0
        A_upper_j[rows, self.pair_j] = -1.0

        A_ub = np.vstack((A_value, A_lower, A_upper_i, A_upper_j))
        b_ub = np.hstack((b_value, np.ones(n_pairs), np.zeros(2 * n_pairs)))

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([float(self.assortment_size)])

        bounds = [(0, 1)] * (n + n_pairs) + [(None, None)]
        integer = [1] * n + [0] * (n_pairs + 1)

        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "integer": integer,
        }

    def lagrange_gradient(self, x_t, state, eq_duals, ineq_duals):
        """Gradient of a node's objective value with respect to theta.

        x_t is the node's LP solution restricted to the assortment variables x.
        The pair variables are recovered as their optimum for this x: y only
        enters the objective and the McCormick constraints, so the LP sets
        y_ij = max(0, x_i + x_j - 1) when q_ij > 0 and y_ij = min(x_i, x_j)
        otherwise, which is x_i x_j for binary x.

        The value-piece weights are lambda_l = -ineq_duals[l] (the solvers report
        nonpositive marginals for <= rows of a minimisation). Without duals, the
        active piece of the assortment is used, which gives the exact gradient of
        Q_theta(s_t, x) for binary x.
        """
        _, w, psi, c, b = self.split_theta()
        phi, phi_bar, phi_pairs = self._features(state)
        x = np.asarray(x_t, dtype=float).reshape(-1)
        x_i, x_j = x[self.pair_i], x[self.pair_j]

        q = phi_pairs @ w
        y = np.where(q > 0, np.maximum(0.0, x_i + x_j - 1.0), np.minimum(x_i, x_j))

        if ineq_duals is not None and len(ineq_duals) >= self.n_value_pieces:
            lam = -np.asarray(ineq_duals[: self.n_value_pieces], dtype=float)
        else:
            d = b + c @ phi_bar
            g = psi @ phi.T
            lam = np.zeros(self.n_value_pieces)
            lam[np.argmax(d + g @ x)] = 1.0

        shown = phi.T @ x
        dQda = -shown
        dQdw = phi_pairs.T @ y
        dQdpieces = np.hstack(
            (
                lam[:, None] * shown[None, :],  # dQ/dpsi_l
                lam[:, None] * phi_bar[None, :],  # dQ/dc_l
                lam[:, None],  # dQ/db_l
            )
        )
        return np.hstack((dQda, dQdw, dQdpieces.reshape(-1)))

    def update_params(self, grad, lr):
        self.theta -= lr * np.asarray(grad, dtype=float)
