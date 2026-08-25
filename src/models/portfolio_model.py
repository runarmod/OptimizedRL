from copy import copy

import numpy as np

from src.models.model_interface import Model


class PortfolioModel(Model):
    def __init__(
        self,
        c,
        C,
        D,
        E,
        aA,
        aB,
        b,
        bounds,
        integer,
        pf,
        exact=False,
        transaction_cost=0.0,
        holding_cost=0.0,
        budget_cap=None,
        initial_cash=30.0,
        risk_cap=None,
        risk_weight=1.0,
        asset_max_position=None,
        action_mode="absolute",
        market_mode="linear",
        return_signal_scale=1.0,
        lr_mult_aA=1.0,
        lr_mult_aB=1.0,
        lr_mult_b=1.0,
        position_dynamics_mode="legacy",
        cvar_mode="off",
        cvar_cap=1.0,
        cvar_alpha=0.95,
        cvar_n_scenarios=20,
        cvar_obj_weight=0.0,
        price_levels_mode="off",
        initial_asset_price=100.0,
    ):
        self.c = c.astype(float)
        self.C = C.astype(float)
        self.D = D.astype(float)
        self.E = E.astype(float)
        self.aA = aA.astype(float)
        self.aB = aB.astype(float)
        self.b = b.astype(float)
        self.bounds = bounds
        self.integer = integer
        self.s_t = None
        self.n_value_pieces = aA.shape[0]
        self.n_desc_vars = self.aB.shape[1]
        self.pf = pf
        self.transaction_cost = float(transaction_cost)
        self.holding_cost = float(holding_cost)
        self.budget_cap = None if budget_cap is None else float(budget_cap)
        self.cash_t = float(initial_cash)
        self.action_mode = str(action_mode)
        self.market_mode = str(market_mode)
        self.return_signal_scale = float(return_signal_scale)
        self.lr_mult_aA = float(lr_mult_aA)
        self.lr_mult_aB = float(lr_mult_aB)
        self.lr_mult_b = float(lr_mult_b)
        self.position_dynamics_mode = str(position_dynamics_mode)
        if self.position_dynamics_mode not in ("legacy", "trade_stateful"):
            raise ValueError(
                "position_dynamics_mode must be 'legacy' or 'trade_stateful'"
            )
        self.risk_cap = None if risk_cap is None else float(risk_cap)
        self.risk_weight = 1.0 if risk_weight is None else float(risk_weight)
        self.asset_max_position = (
            None if asset_max_position is None else float(asset_max_position)
        )
        self.cvar_mode = str(cvar_mode)
        if self.cvar_mode not in ("off", "on"):
            raise ValueError("cvar_mode must be 'off' or 'on'")
        self.cvar_cap = float(cvar_cap)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_n_scenarios = int(cvar_n_scenarios)
        self.cvar_obj_weight = float(cvar_obj_weight)
        self.scenario_matrix = None
        self.price_levels_mode = str(price_levels_mode)
        if self.price_levels_mode not in ("off", "on"):
            raise ValueError("price_levels_mode must be 'off' or 'on'")
        self.initial_asset_price = float(initial_asset_price)
        self.current_prices = self.initial_asset_price * np.ones(
            (self.n_desc_vars,), dtype=float
        )
        self.prev_action = np.zeros((self.n_desc_vars,), dtype=float)

    def get_desc_var_indices(self):
        return slice(self.n_desc_vars)

    def update_state(self, s_t):
        self.s_t = s_t

    def update_prev_action(self, prev_action):
        self.prev_action = np.asarray(prev_action, dtype=float).flatten()

    def update_cash(self, cash_t):
        self.cash_t = float(cash_t)

    def update_scenarios(self, scenario_matrix):
        """Update the rolling return scenario matrix used for CVaR constraint."""
        self.scenario_matrix = np.asarray(scenario_matrix, dtype=float)

    def update_prices(self, prices):
        self.current_prices = np.asarray(prices, dtype=float).flatten()

    def get_LP_formulation(self):
        n_actions = self.n_desc_vars
        unit_prices = np.ones((n_actions,), dtype=float)
        if self.price_levels_mode == "on":
            unit_prices = np.maximum(self.current_prices, 1e-9)
        action_cost = self.c + self.holding_cost * unit_prices
        if self.market_mode == "returns":
            expected_returns = np.asarray(self.s_t[:n_actions], dtype=float)
            action_cost = (
                action_cost - self.return_signal_scale * expected_returns * unit_prices
            )

        if self.position_dynamics_mode == "legacy":
            # Variables: [x (target position), y, s, z]
            c = np.hstack(
                (action_cost, 1, self.pf, self.transaction_cost * unit_prices)
            )
            var_dim = n_actions + 2 + n_actions
            x_start = 0
            y_idx = n_actions
            s_idx = n_actions + 1
            z_start = n_actions + 2
            u_start = None
        else:
            # Variables: [x (target position), y, s, z (|trade|), u (trade)]
            # with equality x - u = prev_position.
            c = np.hstack(
                (
                    action_cost,
                    1,
                    self.pf,
                    self.transaction_cost * unit_prices,
                    np.zeros((n_actions,)),
                )
            )
            var_dim = n_actions + 2 + n_actions + n_actions
            x_start = 0
            y_idx = n_actions
            s_idx = n_actions + 1
            z_start = n_actions + 2
            u_start = n_actions + 2 + n_actions

        neg_ones = -1 * np.ones((self.aA.shape[0], 1))
        A_ub_upper = np.zeros((self.aA.shape[0], var_dim))
        A_ub_upper[:, x_start : x_start + n_actions] = self.aB
        A_ub_upper[:, y_idx : y_idx + 1] = neg_ones
        b_ub_upper = -self.b - self.aA @ self.s_t

        A_ub_lower = np.zeros((self.D.shape[0], var_dim))
        A_ub_lower[:, x_start : x_start + n_actions] = self.D
        A_ub_lower[:, s_idx : s_idx + 1] = -np.ones((self.D.shape[0], 1))
        b_ub_lower = self.E - self.C @ self.s_t

        if self.position_dynamics_mode == "legacy":
            # z_i >= |x_i - prev_action_i|
            A_ub_tx_pos = np.zeros((n_actions, var_dim))
            A_ub_tx_neg = np.zeros((n_actions, var_dim))
            for i in range(n_actions):
                A_ub_tx_pos[i, x_start + i] = 1
                A_ub_tx_pos[i, z_start + i] = -1
                A_ub_tx_neg[i, x_start + i] = -1
                A_ub_tx_neg[i, z_start + i] = -1
            b_ub_tx_pos = self.prev_action.copy()
            b_ub_tx_neg = -self.prev_action.copy()
            A_eq = None
            b_eq = None
        else:
            # z_i >= |u_i|
            A_ub_tx_pos = np.zeros((n_actions, var_dim))
            A_ub_tx_neg = np.zeros((n_actions, var_dim))
            for i in range(n_actions):
                A_ub_tx_pos[i, u_start + i] = 1
                A_ub_tx_pos[i, z_start + i] = -1
                A_ub_tx_neg[i, u_start + i] = -1
                A_ub_tx_neg[i, z_start + i] = -1
            b_ub_tx_pos = np.zeros((n_actions,))
            b_ub_tx_neg = np.zeros((n_actions,))

            # x - u = prev_action
            A_eq = np.zeros((n_actions, var_dim))
            for i in range(n_actions):
                A_eq[i, x_start + i] = 1
                A_eq[i, u_start + i] = -1
            b_eq = self.prev_action.copy()

        A_ub = np.vstack((A_ub_upper, A_ub_lower, A_ub_tx_pos, A_ub_tx_neg))
        b_ub = np.hstack((b_ub_upper, b_ub_lower, b_ub_tx_pos, b_ub_tx_neg))

        if self.budget_cap is not None:
            A_ub_budget = np.zeros((1, var_dim))
            A_ub_budget[0, :n_actions] = unit_prices
            A_ub = np.vstack((A_ub, A_ub_budget))
            b_ub = np.hstack((b_ub, np.array([self.budget_cap])))

        if self.risk_cap is not None:
            A_ub_risk = np.zeros((1, var_dim))
            A_ub_risk[0, :n_actions] = self.risk_weight * unit_prices
            A_ub = np.vstack((A_ub, A_ub_risk))
            b_ub = np.hstack((b_ub, np.array([self.risk_cap])))

        # Self-financing constraint: positions plus immediate friction costs
        # cannot exceed current available cash.
        A_ub_cash = np.zeros((1, var_dim))
        A_ub_cash[0, :n_actions] = (1.0 + self.holding_cost) * unit_prices
        A_ub_cash[0, z_start : z_start + n_actions] = (
            self.transaction_cost * unit_prices
        )
        A_ub = np.vstack((A_ub, A_ub_cash))
        b_ub = np.hstack((b_ub, np.array([self.cash_t])))

        bounds = copy(self.bounds)
        if self.asset_max_position is not None:
            for i in range(n_actions):
                lb, ub = bounds[i]
                new_ub = self.asset_max_position
                if ub is not None:
                    new_ub = min(float(ub), new_ub)
                bounds[i] = (lb, new_ub)
        integer = copy(self.integer)
        bounds.append((None, None))
        bounds.append((0, None))
        for _ in range(n_actions):
            bounds.append((0, None))
        if self.position_dynamics_mode == "trade_stateful":
            for _ in range(n_actions):
                bounds.append((None, None))
        integer.append(0)
        integer.append(0)
        for _ in range(n_actions):
            integer.append(0)
        if self.position_dynamics_mode == "trade_stateful":
            for _ in range(n_actions):
                integer.append(0)

        # CVaR (Rockafellar & Uryasev) risk constraint
        # Variables appended: [eta (VaR level), v_0 .. v_{S-1} (per-scenario excess loss)]
        # Constraint: eta + 1/((1-alpha)*S) * sum(v_s) <= cvar_cap
        # Per-scenario: v_s >= -R_s·x - eta  (v_s >= 0 by bounds)
        if self.cvar_mode == "on" and self.scenario_matrix is not None:
            R = self.scenario_matrix  # shape (S, N)
            S_scen = R.shape[0]
            denom = max((1.0 - self.cvar_alpha) * S_scen, 1e-12)
            cvar_extra = 1 + S_scen  # eta + S v variables
            eta_col = A_ub.shape[1]
            v_start_col = eta_col + 1

            # Extend all existing constraint rows with zero columns
            A_ub = np.hstack((A_ub, np.zeros((A_ub.shape[0], cvar_extra))))
            if A_eq is not None:
                A_eq = np.hstack((A_eq, np.zeros((A_eq.shape[0], cvar_extra))))

            # Extend objective: optional soft penalty on CVaR value
            c = np.hstack(
                (
                    c,
                    np.array([self.cvar_obj_weight]),
                    (self.cvar_obj_weight / denom) * np.ones(S_scen),
                )
            )

            full_dim = A_ub.shape[1]

            # S loss constraints: -R_s·x - eta - v_s <= 0
            A_loss = np.zeros((S_scen, full_dim))
            for s in range(S_scen):
                A_loss[s, :n_actions] = -R[s]
                A_loss[s, eta_col] = -1.0
                A_loss[s, v_start_col + s] = -1.0
            b_loss = np.zeros(S_scen)

            # CVaR cap constraint: eta + (1/denom)*sum(v_s) <= cvar_cap
            A_cap = np.zeros((1, full_dim))
            A_cap[0, eta_col] = 1.0
            A_cap[0, v_start_col : v_start_col + S_scen] = 1.0 / denom
            b_cap = np.array([self.cvar_cap])

            A_ub = np.vstack((A_ub, A_loss, A_cap))
            b_ub = np.hstack((b_ub, b_loss, b_cap))

            # Bounds and integer flags for new variables
            bounds.append((None, None))  # eta: unbounded
            for _ in range(S_scen):
                bounds.append((0, None))  # v_s >= 0
            integer.append(0)  # eta continuous
            for _ in range(S_scen):
                integer.append(0)  # v_s continuous

        node = {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "integer": integer,
        }
        return node

    def lagrange_gradient(self, x_t, state, eq_duals, ineq_duals):
        if ineq_duals is None or len(ineq_duals) < self.n_value_pieces:
            # If solver does not provide duals (e.g., infeasible/fathomed branch),
            # skip parameter update contribution for this sample.
            return np.zeros(
                (self.c.size + self.aA.size + self.aB.size + self.b.size,), dtype=float
            )
        dLdc = x_t
        dLdaA = []
        dLdaB = []
        dLdb = []
        for i in range(self.n_value_pieces):
            dLdaA.append(ineq_duals[i] * state)
            dLdaB.append(ineq_duals[i] * x_t)
            dLdb.append(ineq_duals[i])
        dLdc = np.array(dLdc).flatten()
        dLdaA = -np.array(dLdaA).flatten()
        dLdaB = -np.array(dLdaB).flatten()
        dLdb = -np.array(dLdb).flatten()
        return np.hstack((dLdc, dLdaA, dLdaB, dLdb))

    def get_params(self):
        return super().get_params()

    def update_params(self, grad, lr):
        grad = -grad
        idx = 0

        idx += self.c.size
        self.aA += (lr * self.lr_mult_aA) * grad[idx : idx + self.aA.size].reshape(
            self.aA.shape
        )
        idx += self.aA.size
        self.aB += (lr * self.lr_mult_aB) * grad[idx : idx + self.aB.size].reshape(
            self.aB.shape
        )
        idx += self.aB.size
        self.b += (lr * self.lr_mult_b) * grad[idx:]


Arbbin = PortfolioModel
