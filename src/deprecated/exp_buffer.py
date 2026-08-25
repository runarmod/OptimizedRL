# Taken from https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/deep-q-learning?ex=3

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#     def push(self, state, action, reward, next_state, done):
#         experience_tuple = (state, action, reward, next_state, done)
#         # Append experience_tuple to the memory buffer
#         self.memory.____
#     def __len__(self):
#         return len(self.memory)
#     def sample(self, batch_size):
#         # Draw a random sample of size batch_size
#         batch = ____(____, ____)
#         # Transform batch into a tuple of lists
#         states, actions, rewards, next_states, dones = ____
#         states_tensor = torch.tensor(states, dtype=torch.float32)
#         rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
#         next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
#         dones_tensor = torch.tensor(dones, dtype=torch.float32)
#         # Ensure actions_tensor has shape (batch_size, 1)
#         actions_tensor = torch.tensor(actions, dtype=torch.long).____
#         return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor


