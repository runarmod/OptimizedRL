from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from src.dvsp.simulator import (
    DynamicVSP,
    EpochState,
    routes_cost,
    routes_from_arc_values,
)

GREEDY_PRIZE = 1e9
LAZY_PRIZE = -1e9


@dataclass
class PrizeCollectingLP:
    """Prize-collecting VSP of one epoch, as a minimisation problem.

    min  sum_a (d_a - theta_ext[dst(a)]) y_a
    s.t. inflow(v) = outflow(v)          for every request v
         inflow(v) <= 1                  for every request v
         inflow(v) = 1                   for every must-dispatch request v
         0 <= y_a <= 1

    theta_ext is the prize of postponable requests and 0 elsewhere. This is
    eq. (14) of paper 02 with the objective negated (the framework minimises).
    The constraint matrix is a network matrix, so LP vertices are integral.
    """

    cost: np.ndarray  # arc durations d_a, shape (n_arcs,)
    inflow: sparse.csr_matrix  # (n_locations, n_arcs), 1 if arc enters location
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    postponable: np.ndarray  # location indices that carry a prize

    @classmethod
    def from_state(cls, state: EpochState) -> "PrizeCollectingLP":
        arcs = state.arcs
        n_loc = state.n_locations
        n_arcs = len(arcs)
        cols = np.arange(n_arcs)
        ones = np.ones(n_arcs)
        inflow = sparse.csr_matrix((ones, (arcs[:, 1], cols)), shape=(n_loc, n_arcs))
        outflow = sparse.csr_matrix((ones, (arcs[:, 0], cols)), shape=(n_loc, n_arcs))

        requests = np.arange(1, n_loc)
        must = np.flatnonzero(state.is_must_dispatch)
        flow = (inflow - outflow)[requests]
        A_eq = sparse.vstack((flow, inflow[must])).tocsr()
        b_eq = np.concatenate((np.zeros(len(requests)), np.ones(len(must))))
        A_ub = inflow[requests]
        b_ub = np.ones(len(requests))

        return cls(
            cost=state.duration[arcs[:, 0], arcs[:, 1]],
            inflow=inflow,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            postponable=np.flatnonzero(state.is_postponable),
        )

    @property
    def n_arcs(self) -> int:
        return int(self.cost.shape[0])

    def objective(self, prizes: np.ndarray) -> np.ndarray:
        """Objective vector for the given prizes of the postponable requests."""
        prize_ext = np.zeros(self.inflow.shape[0])
        prize_ext[self.postponable] = prizes
        return self.cost - self.inflow.T @ prize_ext

    def served(self, y: np.ndarray) -> np.ndarray:
        """Inflow of every location, i.e. 1 if the request is dispatched."""
        return self.inflow @ np.asarray(y, dtype=float)


def solve_prize_collecting(
    state: EpochState, prizes: np.ndarray
) -> tuple[list[list[int]], np.ndarray, float]:
    """Solve the prize-collecting VSP with HiGHS dual simplex.

    Returns routes (local indices), the arc vector and the objective value.
    """
    lp = PrizeCollectingLP.from_state(state)
    if lp.n_arcs == 0:
        return [], np.zeros(0), 0.0
    res = linprog(
        lp.objective(np.asarray(prizes, dtype=float)),
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=(0, 1),
        method="highs-ds",
    )
    if not res.success:
        raise RuntimeError(f"prize-collecting LP failed: {res.message}")
    y = np.round(res.x, 6)
    return routes_from_arc_values(state.arcs, y), y, float(res.fun)


def anticipative_solve(
    sim: DynamicVSP, time_limit: float | None = None
) -> tuple[float, list[list[list[int]]]]:
    """Hindsight-optimal dispatching for the revealed requests of ``sim``.

    Port of ``anticipative_solver``. Call ``sim.draw_all_epochs(seed)`` first.
    Returns the total cost and, per epoch, routes in simulator indices.
    """
    n = len(sim.customer_index)
    duration = sim.instance.duration[np.ix_(sim.customer_index, sim.customer_index)]
    start, service = sim.start_time, sim.service_time
    E, delta = sim.epoch_duration, sim.delta_dispatch
    epochs = range(sim.first_epoch, sim.last_epoch + 1)

    def available(i, t):
        return (
            sim.request_epoch[i] <= t
            and (t - 1) * E + duration[0, i] + delta <= start[i]
        )

    compatible = (
        (start[:, None] <= start[None, :])
        & (start[:, None] + service[:, None] + duration <= start[None, :])
    )

    arcs = []  # (i, j, t)
    for t in epochs:
        avail = [i for i in range(1, n) if available(i, t)]
        for i in avail:
            arcs.append((0, i, t))
            arcs.append((i, 0, t))
            for j in avail:
                if i != j and compatible[i, j]:
                    arcs.append((i, j, t))
    arcs = np.asarray(arcs, dtype=np.int64).reshape(-1, 3)
    n_var = len(arcs)
    if n_var == 0:
        return 0.0, [[] for _ in epochs]

    cols = np.arange(n_var)
    t_offset = sim.first_epoch
    n_t = len(epochs)
    # flow conservation per (location, epoch) for customers
    row_in = arcs[:, 1] * n_t + (arcs[:, 2] - t_offset)
    row_out = arcs[:, 0] * n_t + (arcs[:, 2] - t_offset)
    flow = sparse.csr_matrix(
        (np.ones(n_var), (row_in, cols)), shape=(n * n_t, n_var)
    ) - sparse.csr_matrix((np.ones(n_var), (row_out, cols)), shape=(n * n_t, n_var))
    flow = flow[n_t:]  # drop depot rows
    served = sparse.csr_matrix(
        (np.ones(n_var), (arcs[:, 1], cols)), shape=(n, n_var)
    )[1:]

    res = milp(
        c=duration[arcs[:, 0], arcs[:, 1]],
        constraints=[
            LinearConstraint(flow, 0, 0),
            LinearConstraint(served, 1, 1),
        ],
        integrality=np.ones(n_var),
        bounds=Bounds(0, 1),
        options={} if time_limit is None else {"time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"anticipative MILP failed: {res.message}")

    y = np.round(res.x)
    routes_per_epoch = []
    for t in epochs:
        mask = (arcs[:, 2] == t) & (y > 0.5)
        routes_per_epoch.append(
            routes_from_arc_values(arcs[mask][:, :2], np.ones(int(mask.sum())))
        )
    total = sum(routes_cost(r, duration) for r in routes_per_epoch)
    return float(total), routes_per_epoch
