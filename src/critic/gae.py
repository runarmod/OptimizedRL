import warnings

import torch
import torch.nn.functional as F
from torch import nn, optim

from src.critic.critic_interface import Critic


class GAE(Critic): # Why isnt this camelcase
    def __init__(self,dims,lr,df,eps,tau,device):
        super().__init__()
        self.df = df 
        self.eps = eps
        self.dims = dims
        self.device = device
        self.lr = lr
        self.target = DVN(self.dims,device=self.device)
        self.policy = DVN(self.dims,device=self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(),lr = lr) 
        self.tau = tau


    def train(self,rewards,actions,states,nxt_states):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        # nxt_states = torch.tensor(nxt_states)
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
        old_norm = float('inf')
        non_final_mask = torch.tensor(tuple(map(lambda s: torch.any(torch.logical_not( torch.isnan(s) )) ,
                                        nxt_states)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in nxt_states
                                                    if  torch.any(torch.logical_not( torch.isnan(s) ))]
        non_final_next_states = torch.stack(non_final_next_states) if len(non_final_next_states) > 0 else torch.tensor([])
        non_final_next_states = non_final_next_states.float().to(device=self.device)
        state_batch = torch.tensor(states,device=self.device).squeeze().float()
        action_batch = torch.tensor(actions,device=self.device)
        reward_batch = torch.tensor(rewards,device=self.device)
        diff = float('inf')
        while  diff > self.eps:

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy(state_batch).squeeze()

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(len(rewards), device=self.device)
            with torch.no_grad():
                if len(non_final_next_states) > 0 :
                    next_state_values[non_final_mask] = self.target(non_final_next_states).squeeze()
                else:
                    warnings.warn("All states in batch were terminal")
            expected_state_action_values = (next_state_values * self.df) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            self.optimizer.step()
            new_norm = torch.linalg.norm(torch.cat([param.flatten() for param in self.policy.parameters()]))
            diff = abs(new_norm - old_norm)
            old_norm = new_norm


    def reset(self):
        self.target = DVN(self.dims,device=self.device)
        self.policy = DVN(self.dims,device=self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(),lr = self.lr) 
    def evaluate(self, actions, states,rewards,nxt_states):
        non_final_mask = torch.tensor(tuple(map(lambda s: torch.any(torch.logical_not( torch.isnan(s) )) ,
                                        nxt_states)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in nxt_states
                                                    if  torch.any(torch.logical_not( torch.isnan(s) ))]
        non_final_next_states = torch.stack(non_final_next_states) if len(non_final_next_states) > 0 else torch.tensor([])
        non_final_next_states = non_final_next_states.float().to(device=self.device)
        state_batch = torch.tensor(states,device=self.device).squeeze().float()
        action_batch = torch.tensor(actions,device=self.device)
        reward_batch = torch.tensor(rewards,device=self.device)
        with torch.no_grad():

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy(state_batch).squeeze()

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros((len(rewards),), device=self.device)
            if len(non_final_next_states) > 0 :
                next_state_values[non_final_mask] = self.target(non_final_next_states).squeeze() 
            else:
                    warnings.warn("All states in batch were terminal")
            res = (next_state_values * self.df) + reward_batch - state_action_values

        # with torch.no_grad():
        #     res = self.policy(state_batch).cpu().detach().numpy()
        return res.cpu().detach().numpy()
    

class DVN(nn.Module):
    def __init__(self,dims,device):
        super().__init__()
        if dims[-1] != 1:
            raise Exception("Last dim must be one")
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1],device = device) for i in range(len(dims) - 1)])
        
    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)
    

        