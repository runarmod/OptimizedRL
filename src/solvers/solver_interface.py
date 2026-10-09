from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict

import numpy as np

# (lower, upper) per variable; None means unbounded.
type Bounds = list[tuple[float | None, float | None]]

# A branching condition (variable index, ">=" or "<=", value). The variable is a
# SCIP name such as "t_x_3" until it has been parsed to an index.
type BranchCond = tuple[int | str, str, float]


class LPNode(TypedDict):
    """A MILP: minimize c @ x subject to A_ub @ x <= b_ub, A_eq @ x == b_eq, bounds,
    with integrality where integer[i] is truthy."""

    c: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    A_eq: np.ndarray | None
    b_eq: np.ndarray | None
    bounds: Bounds
    integer: list[int]


class Candidate(TypedDict):
    """One entry of a solver's candidate pool: a KKT-valid LP solution."""

    fun: float
    x: np.ndarray
    eqlin: np.ndarray
    ineqlin: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fathomed: bool
    conds: list[BranchCond]
    node: LPNode
    bounds: Bounds
    status: NotRequired[str]
    depth: NotRequired[int | None]
    node_num: NotRequired[int]
    parent: NotRequired[int | None]
    node_status: NotRequired[str]
    branching_decision: NotRequired[dict[str, Any] | None]
    node_bounds: NotRequired[dict[str, dict[str, float]]]
    reduced_costs: NotRequired[dict[str, float]]


class Solver(ABC):
    @abstractmethod
    def solve(self, init_node: LPNode) -> list[Candidate] | None:
        pass
