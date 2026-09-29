from dataclasses import dataclass, field

import numpy as np

from src.dvsp.instance import VSPInstance

N_ACTOR_FEATURES = 14
N_STATE_SUMMARY = 4 + 2 * N_ACTOR_FEATURES + 1
_QUANTILES = np.arange(1, 10) / 10


def create_arcs(start_time, service_time, duration) -> np.ndarray:
    """Arcs of the acyclic VSP graph, as an (n_arcs, 2) array of (src, dst).

    Port of ``create_graph``: the depot connects to and from every customer,
    and customer i precedes customer j when i can be served and j reached by
    j's start time.
    """
    n = len(start_time)
    arcs = []
    for i in range(1, n):
        arcs.append((0, i))
        arcs.append((i, 0))
        for j in range(i + 1, n):
            if start_time[i] <= start_time[j]:
                if start_time[i] + service_time[i] + duration[i, j] <= start_time[j]:
                    arcs.append((i, j))
            elif start_time[j] + service_time[j] + duration[j, i] <= start_time[i]:
                arcs.append((j, i))
    return np.asarray(arcs, dtype=np.int64).reshape(-1, 2)


def routes_cost(routes: list[list[int]], duration: np.ndarray) -> float:
    total = 0.0
    for route in routes:
        current = 0
        for r in route:
            total += duration[current, r]
            current = r
        total += duration[current, 0]
    return float(total)


def routes_from_arc_values(
    arcs: np.ndarray, y: np.ndarray, atol: float = 0.1
) -> list[list[int]]:
    """Decode routes from an arc vector. Port of ``retrieve_routes``."""
    selected = np.abs(np.asarray(y, dtype=float) - 1.0) <= atol
    successor = {}
    starts = []
    for (i, j), used in zip(arcs, selected):
        if not used:
            continue
        if i == 0:
            starts.append(int(j))
        else:
            if i in successor:
                raise ValueError(f"location {i} has more than one successor")
            successor[int(i)] = int(j)

    routes = []
    for start in sorted(starts):
        route = []
        current = start
        while current != 0:
            if current in route or len(route) > len(arcs):
                raise ValueError("arc vector contains a cycle")
            route.append(current)
            if current not in successor:
                raise ValueError(f"route through {current} does not return to depot")
            current = successor[current]
        routes.append(route)
    return routes


def arc_values_from_routes(arcs: np.ndarray, routes: list[list[int]]) -> np.ndarray:
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(arcs)}
    y = np.zeros(len(arcs))
    for route in routes:
        prev = 0
        for r in list(route) + [0]:
            y[index[(prev, r)]] = 1.0
            prev = r
    return y


@dataclass
class EpochState:
    """Decision problem of one epoch (port of ``VSPState``).

    Location 0 is the depot; the rest are the undispatched requests. Start
    times are shifted so that 0 is the planning start time of this epoch.
    """

    epoch: int
    first_epoch: int
    last_epoch: int
    start_time: np.ndarray
    service_time: np.ndarray
    duration: np.ndarray
    is_must_dispatch: np.ndarray
    is_postponable: np.ndarray
    request_index: np.ndarray  # index of each location in the simulator history
    customer_index: np.ndarray  # index of each location in the static instance
    arcs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.arcs = create_arcs(self.start_time, self.service_time, self.duration)

    @property
    def n_locations(self) -> int:
        return int(self.start_time.shape[0])

    @property
    def n_postponable(self) -> int:
        return int(self.is_postponable.sum())

    @property
    def n_must_dispatch(self) -> int:
        return int(self.is_must_dispatch.sum())

    def is_feasible(self, routes: list[list[int]]) -> bool:
        """Port of ``is_feasible``; routes use local (epoch) indices."""
        dispatched = np.zeros(self.n_locations, dtype=bool)
        for route in routes:
            dispatched[route] = True
            current = 0
            time = self.start_time[0]
            for nxt in route:
                time += self.duration[current, nxt]
                if time > self.start_time[nxt] + 1e-9:
                    return False
                time += self.service_time[nxt]
                current = nxt
        if len(set(r for route in routes for r in route)) != sum(map(len, routes)):
            return False
        return bool(np.all(dispatched[self.is_must_dispatch]))


class DynamicVSP:
    """Dynamic VSP simulator. Port of ``DVSPEnv`` in DynamicVehicleRouting.jl.

    Requests are drawn each epoch by sampling locations, start times and
    service times independently from the static instance, and are kept only if
    reachable from the depot (rejection sampling). Request arrivals do not
    depend on the decisions, so the uncertainty is exogenous.
    """

    def __init__(
        self,
        instance: VSPInstance,
        max_requests_per_epoch: int = 10,
        seed: int = 0,
        delta_dispatch: float = 1.0,
        epoch_duration: float = 1.0,
    ):
        self.instance = instance
        self.max_requests_per_epoch = int(max_requests_per_epoch)
        self.seed = seed
        self.delta_dispatch = float(delta_dispatch)
        self.epoch_duration = float(epoch_duration)
        self.first_epoch = 1
        self.last_epoch = (
            int(np.trunc(instance.start_time.max() / self.epoch_duration)) - 1
        )
        self.rng = np.random.default_rng(seed)
        self.state: EpochState | None = None
        self.reset(seed)

    @property
    def n_epochs(self) -> int:
        return self.last_epoch - self.first_epoch + 1

    def reset(self, seed: int | None = None) -> None:
        """Reset the history. A given seed restarts the request stream."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_epoch = self.first_epoch - 1
        self.customer_index = np.array([0], dtype=np.int64)
        self.service_time = np.array([0.0])
        self.start_time = np.array([0.0])
        self.request_is_dispatched = np.array([False])
        self.request_epoch = np.array([self.current_epoch], dtype=np.int64)
        self.state = None

    def is_terminated(self) -> bool:
        return self.current_epoch >= self.last_epoch

    def planning_start_time(self) -> float:
        return (self.current_epoch - 1) * self.epoch_duration + self.delta_dispatch

    def undispatched_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.request_is_dispatched)

    def next_epoch(self) -> EpochState:
        self.current_epoch += 1
        static = self.instance
        n_customers = static.n_customers
        k = min(self.max_requests_per_epoch, n_customers)
        planning_start = self.planning_start_time()

        def sample_indices():
            return self.rng.permutation(n_customers)[:k] + 1

        coordinate_idx = sample_indices()
        start_time_idx = sample_indices()
        service_time_idx = sample_indices()

        feasible = (
            planning_start + static.duration[0, coordinate_idx]
            <= static.start_time[start_time_idx]
        )
        n_new = int(feasible.sum())
        self.customer_index = np.concatenate(
            (self.customer_index, coordinate_idx[feasible])
        )
        self.service_time = np.concatenate(
            (self.service_time, static.service_time[service_time_idx[feasible]])
        )
        self.start_time = np.concatenate(
            (self.start_time, static.start_time[start_time_idx[feasible]])
        )
        self.request_is_dispatched = np.concatenate(
            (self.request_is_dispatched, np.zeros(n_new, dtype=bool))
        )
        self.request_epoch = np.concatenate(
            (self.request_epoch, np.full(n_new, self.current_epoch, dtype=np.int64))
        )
        return self._update_state()

    def _update_state(self) -> EpochState:
        static = self.instance
        planning_start = self.planning_start_time()
        undispatched = self.undispatched_indices()
        customers = self.customer_index[undispatched]

        if self.current_epoch < self.last_epoch:
            is_must = (
                planning_start + self.epoch_duration + static.duration[0, customers]
                > self.start_time[undispatched]
            )
        else:
            is_must = np.ones(len(undispatched), dtype=bool)
        is_must[0] = False
        is_postponable = ~is_must
        is_postponable[0] = False

        self.state = EpochState(
            epoch=self.current_epoch,
            first_epoch=self.first_epoch,
            last_epoch=self.last_epoch,
            start_time=self.start_time[undispatched] - planning_start,
            service_time=self.service_time[undispatched],
            duration=static.duration[np.ix_(customers, customers)],
            is_must_dispatch=is_must,
            is_postponable=is_postponable,
            request_index=undispatched,
            customer_index=customers,
        )
        return self.state

    def apply_routes(self, routes: list[list[int]]) -> float:
        """Dispatch routes given in local (epoch) indices; return their cost."""
        state = self.state
        for route in routes:
            self.request_is_dispatched[state.request_index[route]] = True
        return routes_cost(routes, state.duration)

    def draw_all_epochs(self, seed: int | None = None) -> None:
        """Reveal all requests without dispatching (for the anticipative expert)."""
        self.reset(seed)
        while not self.is_terminated():
            self.next_epoch()

    # --- features -----------------------------------------------------------

    def quantile_time_to_requests(self) -> np.ndarray:
        """Per undispatched location: deciles of its travel time to all customers."""
        rows = self.customer_index[self.undispatched_indices()]
        return np.quantile(self.instance.duration[rows, 1:], _QUANTILES, axis=1).T

    def location_features(self) -> np.ndarray:
        """14 features for every location of the current state (depot included).

        Port of ``compute_model_free_features`` / ``compute_model_aware_features``,
        including the Julia naming quirk where "timeDepotRequest" is the travel
        time from the request to the depot.
        """
        state = self.state
        start = state.start_time
        model_free = np.column_stack(
            (
                start,
                start + state.service_time,
                state.duration[:, 0],
                state.duration[0, :],
                start - self.epoch_duration,
            )
        )
        return np.hstack((model_free, self.quantile_time_to_requests()))

    def actor_features(self) -> np.ndarray:
        """Actor features of the postponable requests, shape (n_postponable, 14)."""
        return self.location_features()[self.state.is_postponable]

    def state_summary(self) -> np.ndarray:
        """Fixed-size state vector for the PPO value network.

        Paper 02 uses a GNN critic over all requests; the framework's critic is
        an MLP, so the variable-size state is summarised by counts, the epoch
        progress and mean features of all / postponable requests.
        """
        state = self.state
        features = self.location_features()
        n_requests = state.n_locations - 1
        scale = float(max(self.max_requests_per_epoch, 1))
        requests = features[1:]
        postponable = features[state.is_postponable]
        empty = np.zeros(N_ACTOR_FEATURES)
        mean_all = requests.mean(axis=0) if len(requests) else empty
        mean_post = postponable.mean(axis=0) if len(postponable) else empty
        must_round_trip = float(
            (state.duration[0, :] + state.duration[:, 0])[state.is_must_dispatch].sum()
        )
        progress = (self.current_epoch - self.first_epoch) / max(self.n_epochs, 1)
        return np.concatenate(
            (
                [
                    progress,
                    n_requests / scale,
                    state.n_must_dispatch / scale,
                    state.n_postponable / scale,
                ],
                mean_all,
                mean_post,
                [must_round_trip],
            )
        ).astype(np.float32)
