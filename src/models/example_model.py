from copy import copy

import numpy as np

from src.models.model_interface import Model


class Arbbin(Model):  # Fix D
    def __init__(self, c, C, D, E, aA, aB, b, bounds, integer, pf, exact=False):
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

    def get_desc_var_indices(self):
        return slice(self.n_desc_vars)

    def update_state(self, s_t):
        self.s_t = s_t

    def get_LP_formulation(self):
        c = np.hstack((self.c, 1, self.pf))
        neg_ones = -1 * np.ones((self.aA.shape[0], 1))
        A_ub_upper = np.hstack((self.aB, neg_ones, np.zeros_like(neg_ones)))
        b_ub_upper = -self.b - self.aA @ self.s_t
        A_ub_lower = np.hstack(
            (self.D, np.zeros((self.D.shape[0], 1)), -np.ones((self.D.shape[0], 1)))
        )
        b_ub_lower = self.E - self.C @ self.s_t
        A_ub = np.vstack((A_ub_upper, A_ub_lower))
        b_ub = np.hstack((b_ub_upper, b_ub_lower))

        A_eq = None
        b_eq = None
        bounds = copy(self.bounds)  # add bounds and integer
        integer = copy(self.integer)
        bounds.append((None, None))
        bounds.append((0, None))
        integer.append(0)
        integer.append(0)

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
        dLdc = x_t  # remember that this is smaller in dim than the c used as a node
        dLdaA = []
        dLdaB = []
        dLdb = []
        for i in range(self.n_value_pieces):
            dLdaA.append(ineq_duals[i] * state)
            dLdaB.append(ineq_duals[i] * x_t)
            dLdb.append(ineq_duals[i])
        dLdc = np.array(dLdc).flatten()  # should be neg
        dLdaA = -np.array(dLdaA).flatten()
        dLdaB = -np.array(dLdaB).flatten()
        dLdb = -np.array(dLdb).flatten()
        # print(dLda)
        res = np.hstack((dLdc, dLdaA, dLdaB, dLdb))

        return res

    def get_params(self):
        return {
            "aA": self.aA.copy(),
            "aB": self.aB.copy(),
            "b": self.b.copy(),
            "c": self.c.copy(),
        }

    def update_params(self, grad, lr):
        # Initialize index tracker
        grad = -grad
        # grad = 0
        idx = 0

        # # Update self.c parameters
        # self.c += lr * grad[idx : idx + self.c.size]
        idx += self.c.size

        # # Extract and reshape gradients for aA
        self.aA += lr * grad[idx : idx + self.aA.size].reshape(self.aA.shape)
        idx += self.aA.size

        # # Extract and reshape gradients for aB
        self.aB += lr * grad[idx : idx + self.aB.size].reshape(self.aB.shape)

        idx += self.aB.size
        self.b += lr * grad[idx:]
