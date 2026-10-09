import numpy as np

# Rows of the feature matrix, counted from the end so they hold for any d
# (the first d rows are the static features).
HYPE, SATISFACTION, PRICE, TIME = -4, -3, -2, -1

# Hidden customer model - assumes d = 2
CUSTOMER_MODEL = np.array([0.3, 0.5, 0.6, -0.4, -0.8, 0.0])


class DynamicAssortmentSimulator:
    def __init__(
        self,
        N: int,
        d: int,
        K: int,
        J: int,
        max_steps: int = 80,
        seed: int = 0,
    ) -> None:
        self.N = N  # Number of items
        assert d == 2, "The CUSTOMER_MODEL only works for d = 2"
        self.d = d  # Dimension of feature vectors (in addition to hype, satisfaction, and price)
        self.K = K  # Assortment size constraint
        self.J = J  # Initial inventory
        self.max_steps = max_steps  # Steps per history
        self.purchase_hist: list[int] = []
        self.current_step = 0
        self.seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed

        rng = np.random.default_rng(self.seed)
        self.features = rng.uniform(1.0, 10.0, (self.d + 3, self.N))
        self.features = np.vstack((self.features, np.ones((1, self.N))))
        self.d_features = np.zeros((2, self.N))
        self.inventory = np.ones(self.N)
        self.purchase_hist = []
        self.current_step = 1

    def hype_update(self) -> np.ndarray:
        hype_vector = np.ones(self.N)

        latest_item = self.purchase_hist[-1]
        if latest_item != -1:
            hype_vector[latest_item] += 0.02

        for offset in range(2, 6):
            if len(self.purchase_hist) < offset:
                break

            previous_item = self.purchase_hist[-offset]
            if previous_item != -1:
                hype_vector[previous_item] -= 0.005

        return hype_vector

    def apply_purchase(self, item: int) -> None:
        old_features = np.copy(self.features)
        self.purchase_hist.append(item)

        hype_vector = self.hype_update()
        self.features[HYPE] *= hype_vector
        if item != -1:
            self.features[SATISFACTION, item] *= 1.01
        self.features[TIME] += 9 / self.max_steps

        self.d_features = self.features[HYPE:PRICE] - old_features[HYPE:PRICE]
        if item != -1:
            self.inventory[item] -= 1 / self.J

        self.inventory = np.round(self.inventory, decimals=4)
        self.current_step += 1

    def customer_utilities(self) -> np.ndarray:
        return CUSTOMER_MODEL @ self.features

    @staticmethod
    def choice_probabilities(utility: np.ndarray, assortment: np.ndarray) -> np.ndarray:
        weight = np.exp(utility) * assortment
        return np.append(weight, 1) / (1 + weight.sum())

    @property
    def prices(self) -> np.ndarray:
        return self.features[PRICE]

    def price(self, item: int) -> float:
        return float(self.prices[item])

    def purchase(self, assortment: np.ndarray) -> tuple[int, float]:
        probabilities = self.choice_probabilities(self.customer_utilities(), assortment)
        rng = np.random.default_rng(self.seed + self.current_step)
        choice_idx = int(rng.choice(self.N + 1, p=probabilities))
        if choice_idx == self.N:
            return -1, 0.0
        return choice_idx, self.price(choice_idx)

    def is_terminated(self) -> bool:
        return (
            self.current_step > self.max_steps
            or int(np.count_nonzero(self.inventory)) < self.K
        )

    def step(self, assortment: np.ndarray) -> tuple[int, float]:
        item, revenue = self.purchase(assortment)
        self.apply_purchase(item)
        return item, revenue
