import gymnasium as gym
import numpy as np


class PortfolioEnv(gym.Env):
    def __init__(
        self,
        c,
        p,
        A,
        B,
        C,
        D,
        E,
        pf,
        a_space_size,
        noise=False,
        std=0,
        transaction_cost=0.0,
        holding_cost=0.0,
        budget_cap=None,
        initial_cash=30.0,
        cash_interest_rate=0.0,
        risk_cap=None,
        risk_weight=1.0,
        asset_max_position=None,
        action_mode="absolute",
        market_mode="linear",
        return_mu=0.0,
        return_phi=0.0,
        return_sigma=0.01,
        alpha_mode="off",
        alpha_rho=0.9,
        alpha_sigma=0.02,
        alpha_to_return=0.2,
        signal_noise_std=0.01,
        reward_mode="economic",
        inventory_penalty=0.0,
        cvar_n_scenarios=20,
        cvar_alpha=0.95,
        price_levels_mode="off",
        initial_asset_price=100.0,
        min_asset_price=1.0,
        seed_behavior="modern",
    ):
        self.c = c
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.pf = pf
        self.p = p
        self.observation_space = gym.spaces.Box(
            low=-10 * np.ones((A.shape[1],)),
            high=10 * np.ones((A.shape[1],)),
        )
        self.init_space = gym.spaces.Box(
            low=np.zeros((A.shape[1],)),
            high=8 * np.ones((A.shape[1],)),
        )
        self.action_space = gym.spaces.MultiDiscrete(
            a_space_size * np.ones((B.shape[1],))
        )
        self.state = None
        self.std = std
        self.transaction_cost = float(transaction_cost)
        self.holding_cost = float(holding_cost)
        self.budget_cap = None if budget_cap is None else float(budget_cap)
        self.initial_cash = float(initial_cash)
        self.cash_interest_rate = float(cash_interest_rate)
        self.market_mode = str(market_mode)
        self.return_mu = float(return_mu)
        self.return_phi = float(return_phi)
        self.return_sigma = float(return_sigma)
        self.alpha_mode = str(alpha_mode)
        if self.alpha_mode not in ("off", "on"):
            raise ValueError("alpha_mode must be 'off' or 'on'")
        self.alpha_rho = float(alpha_rho)
        self.alpha_sigma = float(alpha_sigma)
        self.alpha_to_return = float(alpha_to_return)
        self.signal_noise_std = float(signal_noise_std)
        self.risk_cap = None if risk_cap is None else float(risk_cap)
        self.risk_weight = 1.0 if risk_weight is None else float(risk_weight)
        self.asset_max_position = (
            None if asset_max_position is None else float(asset_max_position)
        )
        self.action_mode = str(action_mode)
        if self.action_mode not in ("absolute", "target"):
            raise ValueError("action_mode must be 'absolute' or 'target'")
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in ("economic", "legacy"):
            raise ValueError("reward_mode must be 'economic' or 'legacy'")
        self.inventory_penalty = float(inventory_penalty)
        self.cvar_n_scenarios = int(cvar_n_scenarios)
        self.cvar_alpha = float(cvar_alpha)
        self.price_levels_mode = str(price_levels_mode)
        if self.price_levels_mode not in ("off", "on"):
            raise ValueError("price_levels_mode must be 'off' or 'on'")
        self.initial_asset_price = float(initial_asset_price)
        self.min_asset_price = float(min_asset_price)
        self.seed_behavior = str(seed_behavior)
        if self.seed_behavior not in ("legacy", "modern"):
            raise ValueError("seed_behavior must be 'legacy' or 'modern'")
        n_assets_init = B.shape[1]
        self.return_window = np.zeros(
            (self.cvar_n_scenarios, n_assets_init), dtype=float
        )
        self.prices = self.initial_asset_price * np.ones((n_assets_init,), dtype=float)
        self.prev_action = np.zeros((B.shape[1],), dtype=float)
        self.current_position = np.zeros((B.shape[1],), dtype=float)
        self.cash = self.initial_cash
        self.wealth = self.initial_cash
        self.alpha_state = np.zeros((B.shape[1],), dtype=float)
        self.t = 1

    def _get_obs(self):
        pass

    def _normal(self, loc, scale, size):
        if self.seed_behavior == "legacy":
            return np.random.normal(loc, scale, size=size)
        return self.np_random.normal(loc, scale, size=size)

    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None and self.seed_behavior == "modern":
            self.init_space.seed(seed)

        self.state = self.init_space.sample()
        while np.any(self.C @ self.state >= self.E):
            self.state = self.init_space.sample()
        if self.market_mode == "returns":
            n_assets = self.B.shape[1]
            self.state = self.state.astype(float)
            if self.alpha_mode == "on":
                self.alpha_state = self._normal(0.0, self.alpha_sigma, size=(n_assets,))
                self.state[:n_assets] = self.alpha_state + self._normal(
                    0.0, self.signal_noise_std, size=(n_assets,)
                )
            else:
                self.state[:n_assets] = self._normal(
                    self.return_mu, self.return_sigma, size=(n_assets,)
                )
        else:
            self.alpha_state = np.zeros((self.B.shape[1],), dtype=float)
        self.prev_action = np.zeros((self.B.shape[1],), dtype=float)
        self.current_position = np.zeros((self.B.shape[1],), dtype=float)
        self.cash = self.initial_cash
        self.wealth = self.initial_cash
        self.prices = self.initial_asset_price * np.ones(
            (self.B.shape[1],), dtype=float
        )
        self.return_window = np.zeros(
            (self.cvar_n_scenarios, self.B.shape[1]), dtype=float
        )
        self.t = 1
        return self.state, None

    def step(self, action, gen_noise=False):
        action = np.asarray(action)
        if np.issubdtype(action.dtype, np.floating):
            rounded_action = np.rint(action).astype(int)
            if np.allclose(action, rounded_action):
                action = rounded_action
        if not self.action_space.contains(action):
            raise Exception("Action does not belong to action space")
        if self.asset_max_position is not None and np.any(
            action > self.asset_max_position + 1e-9
        ):
            raise Exception("Action violates per-asset max position")

        n_assets = self.B.shape[1]
        target_position = action.astype(float)
        executed_trade = target_position - self.current_position
        effective_position = target_position
        current_prices = self.prices.copy()
        noise = self._normal(0, self.std, self.state.shape)
        nxt_state = self.A @ self.state + self.B @ effective_position + noise
        market_pnl = 0.0
        realized_returns = np.zeros((n_assets,), dtype=float)
        observed_signal = np.asarray(self.state[:n_assets], dtype=float)
        if self.market_mode == "returns":
            if self.alpha_mode == "on":
                next_alpha = self.alpha_rho * self.alpha_state + self._normal(
                    0.0, self.alpha_sigma, size=(n_assets,)
                )
                realized_returns = (
                    self.return_mu
                    + self.alpha_to_return * observed_signal
                    + self._normal(0.0, self.return_sigma, size=(n_assets,))
                )
                next_signal = next_alpha + self._normal(
                    0.0, self.signal_noise_std, size=(n_assets,)
                )
                self.alpha_state = next_alpha
                next_returns = next_signal
            else:
                observed_returns = observed_signal
                realized_returns = (
                    self.return_mu
                    + self.return_phi * (observed_returns - self.return_mu)
                    + self._normal(0, self.return_sigma, size=(n_assets,))
                )
                next_returns = realized_returns
            nxt_state = nxt_state.astype(float)
            nxt_state[:n_assets] = next_returns
            if self.price_levels_mode == "on":
                next_prices = np.maximum(
                    current_prices * (1.0 + realized_returns), self.min_asset_price
                )
                market_pnl = float(effective_position @ (next_prices - current_prices))
            else:
                next_prices = current_prices
                market_pnl = float(effective_position @ realized_returns)
        else:
            next_prices = current_prices
        slack = self.C @ self.state + self.D @ effective_position - self.E
        position_sum = float(np.sum(effective_position))
        position_value = float(np.dot(effective_position, current_prices))
        if self.asset_max_position is not None:
            max_position_utilization = float(
                np.max(effective_position / max(self.asset_max_position, 1e-9))
            )
        else:
            max_position_utilization = 0.0
        budget_measure = (
            position_value if self.price_levels_mode == "on" else position_sum
        )
        budget_violation = 0.0
        if self.budget_cap is not None and budget_measure > self.budget_cap + 1e-9:
            budget_violation = float(budget_measure - self.budget_cap)
        budget_utilization = (
            0.0 if self.budget_cap in (None, 0) else budget_measure / self.budget_cap
        )
        risk_measure = (
            position_value if self.price_levels_mode == "on" else position_sum
        )
        risk_exposure = float(self.risk_weight * risk_measure)
        risk_violation = 0.0
        if self.risk_cap is not None and risk_exposure > self.risk_cap + 1e-9:
            risk_violation = float(risk_exposure - self.risk_cap)
        risk_utilization = (
            0.0 if self.risk_cap in (None, 0) else risk_exposure / self.risk_cap
        )
        turnover = float(np.sum(np.abs(executed_trade)))
        turnover_notional = float(np.dot(np.abs(executed_trade), current_prices))
        trade_cost = self.transaction_cost * turnover_notional
        holding_notional = (
            position_value if self.price_levels_mode == "on" else position_sum
        )
        holding_cost_value = self.holding_cost * holding_notional
        buy_notional = float(np.dot(np.clip(executed_trade, 0.0, None), current_prices))
        required_cash = buy_notional + trade_cost + holding_cost_value
        cash_before = float(self.cash)
        wealth_before = float(self.wealth)
        cash_violation = 0.0
        if required_cash > self.cash + 1e-9:
            cash_violation = float(required_cash - self.cash)
        self.t += 1
        penalty = (slack[slack > 0] * self.pf).max() if np.any(slack > 0) else 0
        penalty += self.pf * (budget_violation + risk_violation + cash_violation)
        if self.market_mode == "returns":
            if self.price_levels_mode == "on":
                trade_notional = float(np.dot(executed_trade, current_prices))
                cash_after = float(
                    cash_before * (1.0 + self.cash_interest_rate)
                    - trade_notional
                    - trade_cost
                    - holding_cost_value
                )
                wealth_before = float(
                    cash_before + self.current_position @ current_prices
                )
                wealth_after = float(cash_after + effective_position @ next_prices)
                base_reward = float(wealth_after - wealth_before - penalty)
            else:
                base_reward = market_pnl - penalty
                cash_after = float(self.cash * (1.0 + self.cash_interest_rate))
                wealth_after = float(self.wealth + base_reward)
        else:
            base_reward = self.c @ effective_position + self.p @ nxt_state - penalty
            cash_after = float(self.cash * (1.0 + self.cash_interest_rate))
            wealth_after = float(self.wealth + base_reward)
        inventory_penalty_value = self.inventory_penalty * float(
            np.abs(np.sum(executed_trade))
        )
        economic_reward = float(base_reward - inventory_penalty_value)
        reward = economic_reward
        terminated = False

        old_state = np.array([self.state])
        self.prev_action = effective_position.copy()
        self.current_position = effective_position.copy()
        self.state = nxt_state
        self.prices = next_prices
        # Update CVaR scenario window (FIFO rolling buffer of realized returns)
        self.return_window = np.roll(self.return_window, -1, axis=0)
        self.return_window[-1] = realized_returns
        self_financing_residual = cash_before - required_cash
        self.cash = cash_after
        self.wealth = wealth_after
        action_index = int("".join(map(str, target_position.astype(int))))

        if self.t > 10:
            terminated = True
        if terminated:
            nxt_state = np.array([np.nan] * len(self.state))

        info = {
            "action": action_index,
            "old_state": old_state,
            "new_state": nxt_state,
            "action_mode": self.action_mode,
            "target_position": effective_position.copy(),
            "turnover": turnover,
            "trade_net": float(np.sum(executed_trade)),
            "trade_abs_sum": turnover,
            "turnover_notional": turnover_notional,
            "trade_cost": float(trade_cost),
            "position_sum": position_sum,
            "position_value": position_value,
            "target_position_sum": position_sum,
            "max_position_utilization": max_position_utilization,
            "required_cash": float(required_cash),
            "budget_violation": float(budget_violation),
            "budget_utilization": float(budget_utilization),
            "risk_exposure": risk_exposure,
            "risk_violation": float(risk_violation),
            "risk_utilization": float(risk_utilization),
            "holding_cost": float(holding_cost_value),
            "market_pnl": float(market_pnl),
            "observed_signal_mean": float(np.mean(observed_signal)),
            "realized_return_mean": float(np.mean(realized_returns)),
            "alpha_state_mean": float(np.mean(self.alpha_state)),
            "price_mean": float(np.mean(self.prices)),
            "price_min": float(np.min(self.prices)),
            "price_max": float(np.max(self.prices)),
            "price_levels_mode": self.price_levels_mode,
            "alpha_mode": self.alpha_mode,
            "cash_before": cash_before,
            "cash_after": cash_after,
            "wealth_before": wealth_before,
            "wealth_after": wealth_after,
            "economic_reward": economic_reward,
            "base_reward": float(base_reward),
            "inventory_penalty": float(inventory_penalty_value),
            "self_financing_residual": float(self_financing_residual),
            "cash_violation": float(cash_violation),
            "scenario_matrix": self.return_window.copy(),
        }

        if self.reward_mode == "legacy":
            reward = -reward
        return self.state, reward, terminated, False, info

    def action_to_index(self, action):
        if np.sum(action) == 0:
            return len(action)
        return np.argmax(action)

    def state_to_index(self, state):
        return int("".join(map(str, state.astype(int))), 2)


Arb_binary = PortfolioEnv
