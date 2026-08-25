import time

import numpy as np


# This is completely the wrong way to calculate it
def adv(q_table):
    """

    Args:
        q_table (_type_): nxm matrix of n actions and m states

    Returns:
        _type_: Table of advantages for the given state
    """

    V_s = np.max(q_table, axis=0)
    # print(V_s)
    adv = q_table - V_s
    return adv


def q_update(q_table, rs, lr, df, actions, states, nxt_states):
    """_summary_

    Args:
        q_table (_type_): _description_
        rs (_type_): _description_
        lr (_type_): _description_
        df (_type_): _description_
        actions (_type_): _description_
        states (_type_): _description_
        nxt_states (_type_): _description_

    Returns:
        _type_: _description_
    """
    q_table = q_table.copy()
    for r, action, state, nxt_state in zip(rs, actions, states, nxt_states):
        state = np.round(state).astype(int)
        if all(nxt_state == -1):
            q_table[action, state] = r
        else:
            nxt_state = np.round(nxt_state).numpy().astype(int)
            q_table[action, state] = q_table[action, state] + lr * (
                r + df * np.max(q_table, axis=0)[nxt_state] - q_table[action, state]
            )
    return q_table


def q_update_vec(target_q_table, q_table, rs, lr, df, actions, states, nxt_states):
    q_table = q_table.copy()
    max_next_q = np.max(
        target_q_table[:, nxt_states], axis=0
    )  # Get max Q-value for next states
    q_table[actions, states] += lr * (rs + df * max_next_q - q_table[actions, states])
    return q_table


def train_q_table(
    target_q_table, q_table, rs, lr, df, actions, states, nxt_states, eps=1e-3, mode=0
):
    """_summary_

    Args:
        q_table (_type_): _description_
        rs (_type_): _description_
        lr (_type_): _description_
        df (_type_): _description_
        actions (_type_): _description_
        states (_type_): _description_
        nxt_states (_type_): _description_
        eps (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """

    old = q_table
    new = q_update(q_table, rs, lr, df, actions, states, nxt_states)

    diff = np.sum(np.abs(new - old))
    iter = 0
    # print(diff)
    while diff > eps:
        old = new
        if mode == 1:
            new = q_update_vec(
                target_q_table, old, rs, lr, df, actions, states, nxt_states
            )
        elif mode == 0:
            new = q_update(old, rs, lr, df, actions, states, nxt_states)

        diff = np.sum(np.abs(new - old))
    return new


# q = np.random.uniform(0,1,size=(4,3))
# print(q)


# adv = state_advs(q,None)
# print(adv)


if __name__ == "__main__":
    # Define example parameters
    num_actions = 4
    num_states = 5

    # Initialize a random Q-table
    np.random.seed(42)
    q_table = np.random.rand(num_actions, num_states)
    q_table2 = np.array(q_table)
    q_table2 = np.zeros_like(q_table)

    # Define example rewards, learning rate, discount factor, actions, states, and next states
    rs = np.array([1.0, 0.5, -0.2])
    lr = 0.01
    df = 0.9
    actions = np.array([0, 2, 1])
    states = np.array([1, 3, 4])
    nxt_states = np.array([2, 0, 1])

    # Print original Q-table
    print("Original Q-table:\n", q_table)

    # Apply vectorized Q-update
    max_next_q = np.max(
        q_table[:, nxt_states], axis=0
    )  # Get max Q-value for next states
    q_table[actions, states] += lr * (rs + df * max_next_q - q_table[actions, states])

    # Print updated Q-table
    print("\nUpdated Q-table:\n")
    start = time.time()
    q_test = train_q_table(
        q_table2, rs, lr, df, actions, states, nxt_states, mode=1, eps=0.00001
    )
    print(rs)
    # print(time.time()-start)
    print()
    print(q_test[actions, states])
    # print()
    # print(q_test-q_table2)
