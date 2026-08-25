
import gymnasium as gym
import numpy as np


class Arb_binary(gym.Env):

    def __init__(self,c,p,A,B,C,D,E,pf,a_space_size,noise = False,std = 0):
        self.c = c 
        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.pf = pf
        self.p = p
        self.observation_space = gym.spaces.Box(low = -10*np.ones((A.shape[1],)),high = 10* np.ones((A.shape[1],)))
        self.init_space = gym.spaces.Box(low = np.zeros((A.shape[1],)),high = 8* np.ones((A.shape[1],)))
        self.action_space = gym.spaces.MultiDiscrete(a_space_size * np.ones((B.shape[1],)))
        self.state = None
        self.std = std
        self.t = 1
    def _get_obs(self):
        pass
    def reset(self,seed = None):
        super().reset(seed  = seed)

        self.state = self.init_space.sample()
        while np.any(self.C @ self.state >= self.E):
            self.state = self.init_space.sample()
        # self.state = np.zeros((self.A.shape[1]))
        self.t = 1
        return self.state,None
    
    def step(self,action,gen_noise = False):
        if not self.action_space.contains(action):
            raise Exception("Action does not belong to action space")
        # Noise = ...
        noise = np.random.normal(0,self.std,self.state.shape)
        
        nxt_state = self.A @ self.state + self.B @ action + noise
        slack =  self.C @ self.state + self.D @ action - self.E
        self.t += 1
        penalty = (slack[slack > 0]*self.pf).max() if np.any(slack > 0) else 0
        reward = self.c @ action + self.p @ nxt_state - penalty
        # reward = self.c @ action + self.p @ nxt_state -np.sum(slack[slack > 0]*self.pf)
        # reward = self.c @ action + self.p @ nxt_state
        terminated = False
        # # print(slack)
        # slack_terminated = False
        # if np.any(slack >= 0):
        #     slack_terminated = True
        #     # terminated = True
        #     reward = -np.sum(slack[slack > 0]*self.pf)
        #     if reward > 0:
        #         raise Exception()
        # for comb in product(range(9),repeat = self.B.shape[1]):
        #     # print(comb)
        #     terminated = True
        #     if np.all(self.C @ nxt_state + self.D @ np.array(comb) <= self.E):
        #         terminated = False or slack_terminated
        #         break
        # # if np.all(np.logical_not(action.astype(int))):
        # #     terminated = True
        # if not self.observation_space.contains(nxt_state.astype(np.float32)):
        #     terminated = True
            # nxt_state,_ = self.reset()
            # nxt_state = np.array([9,9,8])
        
        # old_state = int(''.join(map(str, self.state.astype(int))))
        old_state = np.array([self.state])
        # new_state = int(''.join(map(str, nxt_state.astype(int))))
        self.state = nxt_state
        action_index = int(''.join(map(str, action.astype(int))))
        
        # if np.all(np.abs(old_state - nxt_state) < 1e-3):
        #     terminated = True
        if self.t > 10:
            terminated = True
        if terminated:
            # nxt_state = -1 * np.ones_like(self.state)
            nxt_state = np.array([np.nan]*len(self.state))
            # nxt_state = None

        info = {
        "action" : action_index,
        "old_state" : old_state,
        "new_state" : nxt_state
        }
        
        reward = -reward

     
        return self.state,reward,terminated,False,info
    
    def action_to_index(self, action):
        if np.sum(action) == 0:
            return len(action)
        else:
            return np.argmax(action)
    def state_to_index(self,state):
        return int(''.join(map(str, state.astype(int))), 2)
    
    




# c =- np.array([1,2,2,5,1])
# w = np.array([2,3,1,4,1])
# W_max = 10
# action = np.array([1,0,0,0,0])
# g = KnapsackEnv(c,w,W_max,0,0.1)
# g.reset()
# print(g.step(action))