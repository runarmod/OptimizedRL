import numpy as np

from src.dap import milp
from src.dap.simulator import all_assortments
from src.models.model_interface import Model, VectorModelParameters


class DAPModel(Model):
    """CORL MILP policy for the dynamic assortment problem (see src/dap/milp.py).

    Variables are [x_1..x_N, v] (v only with value pieces). In
    ``get_LP_formulation`` the value pieces are rows of A_ub:

        sum_i (psi_j . phi_i) x_i - v <= -(b_j + c_j . phi_bar)

    and the cardinality is the single row of A_eq. The gradient of a node's
    objective follows the CORL paper's eq. (18) with the LP duals; with
    scipy's sign convention (marginals m_j <= 0 for <= rows, lambda = -m):

        dQ/da = -Phi^T x,  dQ/dpsi_j = lambda_j Phi^T x,
        dQ/dc_j = lambda_j phi_bar,  dQ/db_j = lambda_j.
    """

    def __init__(self, theta: np.ndarray, n_pieces: int, k: int = 4,
                 nns_neighbours: int = 3, nns_beta: float = 0.5, seed: int = 0):
        super().__init__()
        self.n_pieces = int(n_pieces)
        self.theta = np.asarray(theta, dtype=float).reshape(milp.n_params(self.n_pieces))
        self.k = k
        self.nns_neighbours = nns_neighbours
        self.nns_beta = nns_beta
        self.rng = np.random.default_rng(seed)
        self.phi = None
        self._assortments = {}

    def update_from_environment(self, environment):
        self._update_state(environment.state)
        self.phi = milp.model_features(environment.item_features())

    @property
    def n_items(self) -> int:
        return self.phi.shape[0]

    def assortments(self) -> np.ndarray:
        key = (self.n_items, self.k)
        if key not in self._assortments:
            self._assortments[key] = all_assortments(*key)
        return self._assortments[key]

    def get_LP_formulation(self):
        n, J = self.n_items, self.n_pieces
        item_cost, slopes, constants = milp.coefficients(self.theta, J, self.phi)
        if J == 0:
            c = item_cost
            A_ub, b_ub = None, None
            A_eq = np.ones((1, n))
        else:
            c = np.append(item_cost, 1.0)
            A_ub = np.hstack((slopes, -np.ones((J, 1))))
            b_ub = -constants
            A_eq = np.append(np.ones(n), 0.0).reshape(1, -1)
        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": np.array([float(self.k)]),
            "bounds": [(0, 1)] * n + ([(None, None)] if J else []),
            "integer": [1] * n + ([0] if J else []),
            "assortment": {"n_items": n, "k": self.k},
        }

    def lagrange_gradient(self, x_t, state, eq_duals, ineq_duals):
        return self.lagrange_gradient_batch(
            np.asarray(x_t, dtype=float)[None, :], state, [eq_duals], [ineq_duals]
        )[0]

    def lagrange_gradient_batch(self, xs, state, eq_duals, ineq_duals):
        """Gradients of all candidates at once, shape (n_candidates, n_params)."""
        xs = np.asarray(xs, dtype=float)[:, : self.n_items]
        phi_x = xs @ self.phi  # (M, F) = Phi^T x per candidate
        blocks = [-phi_x]
        if self.n_pieces:
            lam = np.zeros((len(xs), self.n_pieces))
            for m, duals in enumerate(ineq_duals):
                duals = np.asarray(duals, dtype=float).reshape(-1)
                if duals.size >= self.n_pieces:
                    lam[m] = -duals[: self.n_pieces]
            phi_bar = self.phi.mean(axis=0)
            for j in range(self.n_pieces):
                blocks += [
                    lam[:, j : j + 1] * phi_x,
                    lam[:, j : j + 1] * phi_bar,
                    lam[:, j : j + 1],
                ]
        return np.hstack(blocks)

    def complete_action(self, entry) -> np.ndarray:
        """Turn a candidate into a valid assortment (exactly k items).

        Integral candidates are returned as they are. For a fractional B&B
        node, nearest-neighbour sampling (the CORL paper's NNS-k): among the
        assortments consistent with the node's branching fixings, take the k
        closest to the LP solution in Manhattan distance and sample one with
        probability proportional to exp(-beta * distance).
        """
        x = np.asarray(entry["x"], dtype=float)[: self.n_items]
        if np.allclose(x, np.round(x), atol=1e-6) and round(x.sum()) == self.k:
            return np.round(x)
        assortments = self.assortments()
        consistent = np.ones(len(assortments), dtype=bool)
        # Branching fixings: SCIP-style (var, op, value) conditions, or the
        # enumeration solver's (item, target) pairs (its expressions are the
        # item variables in order).
        fixings = []
        for var, op, value in entry.get("conds") or []:
            if op == ">=" and value >= 0.5:
                fixings.append((var, 1))
            elif op == "<=" and value <= 0.5:
                fixings.append((var, 0))
        fixings += list(entry.get("fixings") or [])
        for var, target in fixings:
            if isinstance(var, (int, np.integer)) and 0 <= var < self.n_items:
                consistent &= assortments[:, var] == target
        candidates = assortments[consistent] if consistent.any() else assortments
        distance = np.abs(candidates - x).sum(axis=1)
        nearest = np.argsort(distance, kind="stable")[: self.nns_neighbours]
        weights = np.exp(-self.nns_beta * distance[nearest])
        pick = self.rng.choice(nearest, p=weights / weights.sum())
        return candidates[pick]

    def get_desc_var_indices(self):
        return slice(0, self.n_items)

    def get_params(self) -> VectorModelParameters:
        return VectorModelParameters(name="theta", values=self.theta.copy())

    def get_policy_params(self) -> np.ndarray:
        return self.theta.astype(np.float32)

    def set_policy_params(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=float).reshape(milp.n_params(self.n_pieces))

    def update_params(self, grad, lr):
        self.theta -= lr * np.asarray(grad, dtype=float)
