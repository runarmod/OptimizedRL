"""The CORL MILP for the DAP and its closed form for a fixed assortment.

Variables: x_i in {0,1} (show item i), y_ij in [0,1] (items i and j are
both shown; only with pairwise terms) and v (value term; only with value
pieces). With item model features phi_i (the 10 actor features plus a
constant), pair features phi_ij = phi_i * phi_j / 10 (elementwise) and the
mean item features phi_bar (the state):

    min   sum_i (-a . phi_i) x_i + sum_{i<j} (w . phi_ij) y_ij + v
    s.t.  sum_i x_i = k
          y_ij >= x_i + x_j - 1,  y_ij <= x_i,  y_ij <= x_j
          v >= b_j + c_j . phi_bar + sum_i (psi_j . phi_i) x_i     j = 1..J

-a . phi_i is the learned immediate revenue of showing item i. The pair term
models interactions between shown items (cannibalisation: under the
multinomial logit, items shown together compete for one purchase); its
McCormick linearisation gives fractional LP relaxations, so branch-and-bound
builds a real tree. The convex piecewise-linear term models diminishing
returns and the value of the next state (CORL eq. 5). Without pairwise terms
and pieces the policy is "show the k highest linear scores", i.e. paper 02's
actor with a top-k layer.

Parameters theta = [a, (psi_1, c_1, b_1), ..., (psi_J, c_J, b_J), w], where
w is only present with pairwise terms.
"""

from itertools import combinations

import numpy as np

N_MODEL_FEATURES = 11  # 10 actor features + constant
PAIR_FEATURE_SCALE = 10.0  # item features lie in [1, 10]; keeps pair features comparable


def model_features(item_features: np.ndarray) -> np.ndarray:
    return np.hstack((item_features, np.ones((item_features.shape[0], 1))))


def item_pairs(n_items: int) -> np.ndarray:
    return np.array(list(combinations(range(n_items), 2)))


def pair_features(phi: np.ndarray) -> np.ndarray:
    pairs = item_pairs(phi.shape[0])
    return phi[pairs[:, 0]] * phi[pairs[:, 1]] / PAIR_FEATURE_SCALE


def n_params(n_pieces: int, pairwise: bool = False) -> int:
    f = N_MODEL_FEATURES
    return f + n_pieces * (2 * f + 1) + (f if pairwise else 0)


def unpack(theta: np.ndarray, n_pieces: int, pairwise: bool = False):
    f = N_MODEL_FEATURES
    theta = np.asarray(theta, dtype=float)
    a = theta[:f]
    end = f + n_pieces * (2 * f + 1)
    pieces = theta[f:end].reshape(n_pieces, 2 * f + 1) if n_pieces else np.zeros((0, 2 * f + 1))
    psi, c, b = pieces[:, :f], pieces[:, f:2 * f], pieces[:, 2 * f]
    w = theta[end:end + f] if pairwise else None
    return a, psi, c, b, w


def coefficients(theta, n_pieces, phi, pairwise: bool = False):
    """Item costs -a.phi_i, piece slopes psi_j.phi_i, piece constants and pair costs."""
    a, psi, c, b, w = unpack(theta, n_pieces, pairwise)
    item_cost = -(phi @ a)  # (n_items,)
    slopes = psi @ phi.T  # (J, n_items)
    constants = b + c @ phi.mean(axis=0)  # (J,)
    pair_cost = pair_features(phi) @ w if pairwise else None  # (n_pairs,)
    return item_cost, slopes, constants, pair_cost


def assortment_values(theta, n_pieces, phi, assortments, pairwise: bool = False):
    """Q of every assortment (rows of a 0/1 matrix) and its active piece (-1 if none)."""
    item_cost, slopes, constants, pair_cost = coefficients(theta, n_pieces, phi, pairwise)
    q = assortments @ item_cost
    if pairwise:
        # sum_{i<j} q_ij x_i x_j = x^T Q x / 2 with Q symmetric (zero diagonal).
        n = phi.shape[0]
        pairs = item_pairs(n)
        pair_matrix = np.zeros((n, n))
        pair_matrix[pairs[:, 0], pairs[:, 1]] = pair_cost
        pair_matrix += pair_matrix.T
        q = q + ((assortments @ pair_matrix) * assortments).sum(axis=1) / 2
    if n_pieces == 0:
        return q, np.full(len(assortments), -1)
    piece_values = assortments @ slopes.T + constants  # (M, J)
    active = piece_values.argmax(axis=1)
    return q + piece_values[np.arange(len(assortments)), active], active


def initial_theta(n_pieces: int, rng: np.random.Generator, pairwise: bool = False) -> np.ndarray:
    """Glorot-uniform initialisation of every parameter block."""
    f = N_MODEL_FEATURES
    limit = np.sqrt(6.0 / (f + 1))
    return rng.uniform(-limit, limit, n_params(n_pieces, pairwise))
