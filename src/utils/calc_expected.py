import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, identity, lil_matrix, vstack


def formulate_lp_with_initial_state(c, A, B, D, E, F, T, s_initial):
    """
    Formulates the time-dependent LP into standard form min c'z s.t.
    A_eq z = b_eq, A_ub z <= b_ub, assuming a fixed initial state s_0.

    Args:
        c (np.ndarray): Cost vector for x_t (dim n).
        A (np.ndarray or sparse matrix): State transition matrix for s_t (dim m x m).
        B (np.ndarray or sparse matrix): State transition matrix for x_t (dim m x n).
        D (np.ndarray or sparse matrix): Inequality matrix for s_t (dim k x m).
        E (np.ndarray or sparse matrix): Inequality matrix for x_t (dim k x n).
        F (np.ndarray): Right-hand side for inequality constraints (dim k).
        T (int): Time horizon (number of steps, x_t goes from 0 to T-1).
        s_initial (np.ndarray): The fixed initial state vector s_0 (dim m).

    Returns:
        tuple: (c_agg, A_eq, b_eq, A_ub, b_ub)
               Ready for scipy.optimize.linprog (bounds need to be added separately).
               Matrices A_eq and A_ub are returned as CSR sparse matrices.
    """
    # Ensure inputs are numpy arrays for shape info
    c = np.asarray(c)
    s_initial = np.asarray(s_initial)
    F = np.asarray(F)

    # --- Dimensions ---
    n = B.shape[1]  # Dimension of x_t
    m = A.shape[0]  # Dimension of s_t
    if D is not None and E is not None:
        k = D.shape[0]  # Number of inequality constraints per step
    else:  # Handle case with no inequality constraints D, E, F
        k = 0

    if s_initial.shape[0] != m:
        raise ValueError(
            f"s_initial dimension ({s_initial.shape[0]}) must match A rows ({m})"
        )
    if c.shape[0] != n:
        raise ValueError(f"c dimension ({c.shape[0]}) must match B columns ({n})")
    if k > 0 and F.shape[0] != k:
        raise ValueError(f"F dimension ({F.shape[0]}) must match D rows ({k})")

    N = T * n + (T + 1) * m  # Total number of variables in z

    # --- Aggregated Cost Vector c_agg ---
    c_agg_x = np.tile(c, T)
    c_agg_s = np.zeros((T + 1) * m)
    c_agg = np.hstack([c_agg_x, c_agg_s])

    # --- Equality Constraints (Dynamics) A_eq_dynamics z = 0 ---
    num_eq_dynamics = T * m
    A_eq_dynamics = lil_matrix((num_eq_dynamics, N))
    I_m = identity(m, format="csr")  # Use sparse identity

    for t in range(T):
        row_start = t * m
        row_end = (t + 1) * m

        col_start_xt = t * n
        col_end_xt = (t + 1) * n

        col_start_st = T * n + t * m
        col_end_st = T * n + (t + 1) * m

        col_start_st1 = T * n + (t + 1) * m
        col_end_st1 = T * n + (t + 2) * m

        A_eq_dynamics[row_start:row_end, col_start_xt:col_end_xt] = B
        A_eq_dynamics[row_start:row_end, col_start_st:col_end_st] = A
        A_eq_dynamics[row_start:row_end, col_start_st1:col_end_st1] = -I_m

    b_eq_dynamics = np.zeros(num_eq_dynamics)

    # --- Equality Constraints (Initial State) A_eq_s0 z = s_initial ---
    num_eq_s0 = m
    A_eq_s0 = lil_matrix((num_eq_s0, N))
    s0_col_start = T * n  # Column index where s_0 variables begin
    s0_col_end = T * n + m
    A_eq_s0[:, s0_col_start:s0_col_end] = I_m

    b_eq_s0 = s_initial  # RHS is the fixed initial state

    # --- Combine Equality Constraints ---
    A_eq = vstack([A_eq_dynamics, A_eq_s0], format="csr")
    b_eq = np.concatenate([b_eq_dynamics, b_eq_s0])

    # --- Inequality Constraints A_ub z <= b_ub ---
    if k > 0:
        num_ineq = T * k
        A_ub_x = block_diag([E] * T, format="csr")  # Size (T*k) x (T*n)
        A_ub_s_main = block_diag(
            [D] * T, format="csr"
        )  # Size (T*k) x (T*m) (for s_0 to s_{T-1})
        A_ub_s_T_zeros = csr_matrix(
            (num_ineq, m)
        )  # Zero block for s_T columns, Size (T*k) x m
        A_ub_s = hstack(
            [A_ub_s_main, A_ub_s_T_zeros], format="csr"
        )  # Size (T*k) x ((T+1)*m)
        A_ub = hstack([A_ub_x, A_ub_s], format="csr")  # Size (T*k) x N

        if F.ndim > 1:
            F_flat = F.flatten()
        else:
            F_flat = F
        b_ub = np.tile(F_flat, T)
    else:  # No inequality constraints
        # Create empty structures as placeholders or handle as needed by solver
        A_ub = None  # Or csr_matrix((0, N)) depending on solver needs
        b_ub = None  # Or np.array([])

    return c_agg, A_eq, b_eq, A_ub, b_ub


def calc_expected_reward(c, A, B, C, D, E, T, state, solver):
    sols = None
    while sols is None and T > 0:
        c_agg, A_eq, b_eq, A_ub, b_ub = formulate_lp_with_initial_state(
            c, A, B, C, D, E, T, state
        )

        bounds_agg = [(0, 8) for _ in range(A_ub.shape[1])]
        integer_actions = [1 for _ in range(c.size * T)]
        integer_states = [0 for _ in range(A_ub.shape[1] - len(integer_actions))]
        integer_agg = integer_actions + integer_states
        node = {
            "c": c_agg,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds_agg,
            "integer": integer_agg,
        }

        sols = solver.solve(node)
        if sols is None:
            T -= 1
        else:
            best = sols[0]
            for sol in sols:
                if sol["fun"] < best["fun"]:
                    best = sol
            return best["fun"]
