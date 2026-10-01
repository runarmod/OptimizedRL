from itertools import combinations

import numpy as np

# Rows of the feature matrix (paper 02, 1-based rows 1-6).
STATIC_1, STATIC_2, HYPE, SATISFACTION, PRICE, TIME = range(6)
N_ITEM_FEATURES = 10  # actor input per item: 6 features + 2 one-step + 2 cumulative changes

# Hidden customer model: utility = CUSTOMER_WEIGHTS . features[0:6, i]
CUSTOMER_WEIGHTS = np.array([0.3, 0.5, 0.6, -0.4, -0.8, 0.0])


def all_assortments(n_items: int, k: int) -> np.ndarray:
    """0/1 matrix with one row per assortment of size k (4845 rows for 20/4)."""
    combos = np.array(list(combinations(range(n_items), k)))
    out = np.zeros((len(combos), n_items))
    np.put_along_axis(out, combos, 1.0, axis=1)
    return out


def choice_probabilities(utility: np.ndarray, assortment: np.ndarray) -> np.ndarray:
    """Multinomial-logit purchase probabilities; the last entry is no purchase."""
    weights = np.exp(utility) * assortment
    denominator = 1.0 + weights.sum()
    return np.append(weights / denominator, 1.0 / denominator)


class DAPSimulator:
    """Port of the ``DAP`` environment of paper 02.

    Each step shows an assortment of ``k`` of ``n_items`` items. One customer
    buys at most one item according to a multinomial logit on the hidden
    linear utility; the reward is the price paid. Purchases change the
    features: the bought item's hype rises by 2% for one step and decays over
    the next four, and its satisfaction rises permanently by 1%, which lowers
    its future utility. Episodes last ``max_steps`` steps.

    One difference to the Julia code: the purchase history is reset with the
    episode. In the original it persists across episodes, so the first steps'
    hype depends on the previously evaluated episode.
    """

    def __init__(self, n_items=20, n_static=2, k=4, inventory_units=200, max_steps=80):
        self.n_items = n_items
        self.n_static = n_static
        self.k = k
        self.inventory_units = inventory_units
        self.max_steps = max_steps
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        random_rows = rng.uniform(1.0, 10.0, size=(self.n_static + 3, self.n_items))
        self.features = np.vstack((random_rows, np.ones((1, self.n_items))))
        self.start_features = self.features.copy()
        self.d_features = np.zeros((2, self.n_items))
        self.inventory = np.ones(self.n_items)
        self.purchase_hist: list[int] = []
        self.step_count = 1  # 1-based, as in the Julia code

    @property
    def prices(self) -> np.ndarray:
        return self.features[PRICE]

    def utilities(self) -> np.ndarray:
        """True customer utilities (hidden from the agent)."""
        return CUSTOMER_WEIGHTS @ self.features

    def item_features(self) -> np.ndarray:
        """The 10 actor features per item, shape (n_items, 10).

        Paper 02's ``feature_vector``: the 6 feature rows, the one-step change
        of hype and satisfaction, and their change since the episode start.
        """
        delta = self.features[HYPE:SATISFACTION + 1] - self.start_features[HYPE:SATISFACTION + 1]
        return np.vstack((self.features, self.d_features, delta)).T

    def is_terminated(self) -> bool:
        return (
            self.step_count > self.max_steps
            or np.count_nonzero(self.inventory) < self.k
        )

    def purchase(self, assortment: np.ndarray) -> tuple[int, float]:
        """Sample the customer's choice; returns (item or -1, revenue)."""
        probs = choice_probabilities(self.utilities(), assortment)
        rng = np.random.default_rng(self.seed + self.step_count)  # paper: seed + step
        choice = int(rng.choice(self.n_items + 1, p=probs / probs.sum()))
        if choice == self.n_items:
            return -1, 0.0
        return choice, float(self.prices[choice])

    def _hype_vector(self) -> np.ndarray:
        hype = np.ones(self.n_items)
        if self.purchase_hist[-1] >= 0:
            hype[self.purchase_hist[-1]] += 0.02
        for lag in range(2, 6):
            if len(self.purchase_hist) >= lag and self.purchase_hist[-lag] >= 0:
                hype[self.purchase_hist[-lag]] -= 0.005
        return hype

    def apply_purchase(self, item: int) -> None:
        """Port of ``step!``: update features after the customer's choice."""
        old = self.features.copy()
        self.purchase_hist.append(item)
        self.features[HYPE] *= self._hype_vector()
        if item >= 0:
            self.features[SATISFACTION, item] *= 1.01
        self.features[TIME] += 9 / self.max_steps
        self.d_features = self.features[HYPE:SATISFACTION + 1] - old[HYPE:SATISFACTION + 1]
        if item >= 0:
            self.inventory[item] -= 1 / self.inventory_units
        self.inventory = np.round(self.inventory, 4)
        self.step_count += 1

    def step(self, assortment: np.ndarray) -> tuple[int, float]:
        item, revenue = self.purchase(assortment)
        self.apply_purchase(item)
        return item, revenue
