import numpy as np

import src.utils.q_table as q
from critic.critic_interface import Critic


class Q_table(Critic):
    def __init__(self, table, lr, df, eps):
        super().__init__()
        self.table = table
        self.lr = lr
        self.df = df
        self.eps = eps

    def train(self, rewards, actions, states, nxt_states):
        target = self.table.copy()
        self.table = q.train_q_table(
            target,
            self.table,
            rewards,
            self.lr,
            self.df,
            actions,
            states,
            nxt_states,
            self.eps,
            mode=0,
        )
        return self.table

    def evaluate(self, actions, states):
        states = np.round(states).astype(int)
        return self.table[actions, states]
