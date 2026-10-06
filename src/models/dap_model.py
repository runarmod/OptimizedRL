import numpy as np

from src.dap import milp
from src.dap.simulator import all_assortments
from src.models.model_interface import Model, VectorModelParameters


class DAPModel(Model):
    """CORL MILP policy for the dynamic assortment problem (see src/dap/milp.py).

    Variables are [x_1..x_N, y_1..y_P, v]: items, item pairs (only with
    ``pairwise``) and the value term (only with value pieces). Rows of A_ub
    are the J value pieces first,

        sum_i (psi_j . phi_i) x_i - v <= -(b_j + c_j . phi_bar),

    then the McCormick rows of every pair (x_i + x_j - y <= 1, y - x_i <= 0,
    y - x_j <= 0). The cardinality is the single row of A_eq. The gradient of
    a node's objective follows the CORL paper's eq. (18) with the LP duals;
    with scipy's sign convention (marginals m_j <= 0 for <= rows, lambda = -m):

        dQ/da = -Phi^T x,   dQ/dw = sum_ij y_ij phi_ij,
        dQ/dpsi_j = lambda_j Phi^T x,  dQ/dc_j = lambda_j phi_bar,  dQ/db_j = lambda_j.

    The McCormick rows do not depend on the parameters, so their duals are
    not needed.
    """

    def __init__(self, theta: np.ndarray, n_pieces: int, pairwise: bool = False, k: int = 4,
                 nns_neighbours: int = 3, nns_beta: float = 0.5, seed: int = 0,
                 node_values: str = "lp_bound"):
        super().__init__()
        if node_values not in ("lp_bound", "completed"):
            raise ValueError("node_values must be 'lp_bound' or 'completed'")
        self.node_values = node_values
        self.n_pieces = int(n_pieces)
        self.pairwise = bool(pairwise)
        self.theta = np.asarray(theta, dtype=float).reshape(
            milp.n_params(self.n_pieces, self.pairwise)
        )
        self.k = k
        self.nns_neighbours = nns_neighbours
        self.nns_beta = nns_beta
        self.rng = np.random.default_rng(seed)
        self.phi = None
        self._cache = {}

    def update_from_environment(self, environment):
        self._update_state(environment.state)
        self.phi = milp.model_features(environment.item_features())
        self.phi_pairs = milp.pair_features(self.phi) if self.pairwise else None

    @property
    def n_items(self) -> int:
        return self.phi.shape[0]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs()) if self.pairwise else 0

    def pairs(self) -> np.ndarray:
        if ("pairs", self.n_items) not in self._cache:
            self._cache[("pairs", self.n_items)] = milp.item_pairs(self.n_items)
        return self._cache[("pairs", self.n_items)]

    def assortments(self) -> np.ndarray:
        key = ("assortments", self.n_items, self.k)
        if key not in self._cache:
            self._cache[key] = all_assortments(self.n_items, self.k)
        return self._cache[key]

    def _mccormick_rows(self) -> tuple[np.ndarray, np.ndarray]:
        key = ("mccormick", self.n_items, self.pairwise, self.n_pieces)
        if key not in self._cache:
            n, P, J = self.n_items, self.n_pairs, self.n_pieces
            rows = np.zeros((3 * P, n + P + (1 if J else 0)))
            for p, (i, j) in enumerate(self.pairs()):
                rows[3 * p, [i, j, n + p]] = (1, 1, -1)  # x_i + x_j - y <= 1
                rows[3 * p + 1, [n + p, i]] = (1, -1)  # y - x_i <= 0
                rows[3 * p + 2, [n + p, j]] = (1, -1)  # y - x_j <= 0
            self._cache[key] = (rows, np.tile([1.0, 0.0, 0.0], P))
        return self._cache[key]

    def get_LP_formulation(self):
        n, P, J = self.n_items, self.n_pairs, self.n_pieces
        item_cost, slopes, constants, pair_cost = milp.coefficients(
            self.theta, J, self.phi, self.pairwise
        )
        n_vars = n + P + (1 if J else 0)
        c = np.concatenate((item_cost, pair_cost if P else [], [1.0] if J else []))
        ub_rows, ub_rhs = [], []
        if J:
            piece_rows = np.zeros((J, n_vars))
            piece_rows[:, :n] = slopes
            piece_rows[:, -1] = -1.0
            ub_rows.append(piece_rows)
            ub_rhs.append(-constants)
        if P:
            rows, rhs = self._mccormick_rows()
            ub_rows.append(rows)
            ub_rhs.append(rhs)
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        return {
            "c": c,
            "A_ub": np.vstack(ub_rows) if ub_rows else None,
            "b_ub": np.concatenate(ub_rhs) if ub_rhs else None,
            "A_eq": A_eq,
            "b_eq": np.array([float(self.k)]),
            "bounds": [(0, 1)] * (n + P) + ([(None, None)] if J else []),
            "integer": [1] * n + [0] * P + ([0] if J else []),
            "assortment": {"n_items": n, "k": self.k, "evaluate": self.evaluate_assortments},
        }

    def evaluate_assortments(self, assortments: np.ndarray):
        """Closed form for fixed assortments, for the exact solver.

        Returns Q per assortment, the full variable vectors and the duals of
        the value pieces (-1 on the active piece, scipy's sign convention).
        """
        values, active = milp.assortment_values(
            self.theta, self.n_pieces, self.phi, assortments, self.pairwise
        )
        blocks = [assortments]
        if self.pairwise:
            pairs = self.pairs()
            if len(assortments) == len(self.assortments()):
                # The full set of assortments is the same at every step: cache it.
                key = ("pair_products_all", assortments.shape)
                if key not in self._cache:
                    self._cache[key] = assortments[:, pairs[:, 0]] * assortments[:, pairs[:, 1]]
                blocks.append(self._cache[key])
            else:
                blocks.append(assortments[:, pairs[:, 0]] * assortments[:, pairs[:, 1]])
        duals = np.zeros((len(assortments), self.n_pieces))
        if self.n_pieces:
            _, slopes, constants, _ = milp.coefficients(self.theta, self.n_pieces, self.phi)
            v = (assortments @ slopes.T + constants)[np.arange(len(assortments)), active]
            blocks.append(v[:, None])
            duals[np.arange(len(assortments)), active] = -1.0
        return values, np.hstack(blocks), duals

    def lagrange_gradient(self, x_t, state, eq_duals, ineq_duals):
        return self.lagrange_gradient_batch(
            np.asarray(x_t, dtype=float)[None, :], state, [eq_duals], [ineq_duals]
        )[0]

    def lagrange_gradient_batch(self, xs, state, eq_duals, ineq_duals):
        """Gradients of all candidates at once, shape (n_candidates, n_params)."""
        xs = np.asarray(xs, dtype=float)
        x = xs[:, : self.n_items]
        phi_x = x @ self.phi  # (M, F) = Phi^T x per candidate
        blocks = [-phi_x]
        if self.n_pieces:
            lam = np.zeros((len(xs), self.n_pieces))
            for m, duals in enumerate(ineq_duals):
                duals = np.asarray(duals, dtype=float).reshape(-1)
                if duals.size >= self.n_pieces:
                    lam[m] = -duals[: self.n_pieces]
            phi_bar = self.phi.mean(axis=0)
            for j in range(self.n_pieces):
                blocks += [lam[:, j : j + 1] * phi_x, lam[:, j : j + 1] * phi_bar, lam[:, j : j + 1]]
        if self.pairwise:
            if xs.shape[1] >= self.n_items + self.n_pairs:
                y = xs[:, self.n_items : self.n_items + self.n_pairs]
            else:
                pairs = self.pairs()
                y = x[:, pairs[:, 0]] * x[:, pairs[:, 1]]
            blocks.append(y @ self.phi_pairs)
        return np.hstack(blocks)

    def refine_pool(self, pool):
        """Score B&B candidates by the assortment they execute (bias correction).

        With ``node_values="lp_bound"`` the pool is returned unchanged: pruned
        nodes keep their LP bound, as in the CORL paper's eq. (12). That bound
        is optimistic, and with a weak relaxation it says little about the
        assortment NNS-k finally executes. With ``"completed"``, every
        candidate is first completed to an assortment (integral leaves as they
        are, pruned nodes by NNS-k), duplicates are merged, and each is scored
        by its exact Q, with the gradient taken at that assortment. The B&B
        tree still decides which assortments are candidates.
        """
        if self.node_values == "lp_bound" or not pool:
            return pool
        completed, sources, seen = [], [], set()
        for entry in pool:
            action = self.complete_action(entry)
            key = tuple(action.astype(int))
            if key in seen:
                continue
            seen.add(key)
            completed.append(action)
            sources.append(entry)
        values, xs, duals = self.evaluate_assortments(np.array(completed))
        return [
            {
                "fun": float(values[m]),
                "x": xs[m],
                "ineqlin": duals[m],
                "eqlin": np.zeros(1),
                "fathomed": False,
                "conds": source.get("conds", []),
                "node": source.get("node"),
                "bounds": source.get("bounds"),
                "lp_bound": source.get("fun"),
                "status": f"completed_{source.get('status')}",
            }
            for m, source in enumerate(sources)
        ]

    def complete_action(self, entry) -> np.ndarray:
        """Turn a candidate into a valid assortment (exactly k items).

        Integral candidates are returned as they are. For a fractional B&B
        node, nearest-neighbour sampling (the CORL paper's NNS-k): among the
        assortments consistent with the node's branching decisions, take the
        k closest to the LP solution in Manhattan distance and sample one with
        probability proportional to exp(-beta * distance).
        """
        x = np.asarray(entry["x"], dtype=float)[: self.n_items]
        if np.allclose(x, np.round(x), atol=1e-6) and round(x.sum()) == self.k:
            return np.round(x)
        assortments = self.assortments()
        consistent = np.ones(len(assortments), dtype=bool)
        for var, op, value in entry.get("conds") or []:
            if isinstance(var, (int, np.integer)) and 0 <= var < self.n_items:
                if op == ">=" and value >= 0.5:
                    consistent &= assortments[:, var] == 1
                elif op == "<=" and value <= 0.5:
                    consistent &= assortments[:, var] == 0
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
        self.theta = np.asarray(theta, dtype=float).reshape(
            milp.n_params(self.n_pieces, self.pairwise)
        )

    def update_params(self, grad, lr):
        self.theta -= lr * np.asarray(grad, dtype=float)
