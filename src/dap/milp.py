"""The CORL MILP for the DAP and its closed form for a fixed assortment.

Variables: x_i in {0,1} (show item i) and v (value term). With item model
features phi_i (the 10 actor features plus a constant) and their mean
phi_bar over all items (the state):

    min   sum_i (-a . phi_i) x_i + v
    s.t.  sum_i x_i = k
          v >= b_j + c_j . phi_bar + sum_i (psi_j . phi_i) x_i     j = 1..J

-a . phi_i is the learned immediate revenue of showing item i, and the
convex piecewise-linear term models diminishing returns between the shown
items and the future value of the next state (CORL eq. 5). With J = 0 the
value term disappears and the policy is "show the k highest linear scores",
i.e. paper 02's actor with a top-k layer.

Parameters theta = [a, (psi_1, c_1, b_1), ..., (psi_J, c_J, b_J)].
"""

import numpy as np

N_MODEL_FEATURES = 11  # 10 actor features + constant


def model_features(item_features: np.ndarray) -> np.ndarray:
    return np.hstack((item_features, np.ones((item_features.shape[0], 1))))


def n_params(n_pieces: int) -> int:
    f = N_MODEL_FEATURES
    return f + n_pieces * (2 * f + 1)


def unpack(theta: np.ndarray, n_pieces: int):
    f = N_MODEL_FEATURES
    theta = np.asarray(theta, dtype=float)
    a = theta[:f]
    pieces = theta[f:].reshape(n_pieces, 2 * f + 1) if n_pieces else np.zeros((0, 2 * f + 1))
    psi, c, b = pieces[:, :f], pieces[:, f:2 * f], pieces[:, 2 * f]
    return a, psi, c, b


def coefficients(theta, n_pieces, phi):
    """Item scores -a.phi_i, piece slopes psi_j.phi_i and piece constants."""
    a, psi, c, b = unpack(theta, n_pieces)
    item_cost = -(phi @ a)  # (n_items,)
    slopes = psi @ phi.T  # (J, n_items)
    constants = b + c @ phi.mean(axis=0)  # (J,)
    return item_cost, slopes, constants


def assortment_values(theta, n_pieces, phi, assortments):
    """Q of every assortment (rows of a 0/1 matrix) and its active piece."""
    item_cost, slopes, constants = coefficients(theta, n_pieces, phi)
    q = assortments @ item_cost
    if n_pieces == 0:
        return q, np.full(len(assortments), -1)
    piece_values = assortments @ slopes.T + constants  # (M, J)
    active = piece_values.argmax(axis=1)
    return q + piece_values[np.arange(len(assortments)), active], active


def initial_theta(n_pieces: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot-uniform initialisation of every parameter block."""
    f = N_MODEL_FEATURES
    limit = np.sqrt(6.0 / (f + 1))
    return rng.uniform(-limit, limit, n_params(n_pieces))
