import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, num_agents, state_sizes, action_sizes):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((buffer_size, num_agents, max(state_sizes)), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, max(action_sizes)), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, num_agents, max(state_sizes)), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)
    
    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.states[self.ptr, i, :len(states[i])] = states[i]
            self.actions[self.ptr, i, :len(actions[i])] = actions[i]
            self.next_states[self.ptr, i, :len(next_states[i])] = next_states[i]
        
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        # Extract samples
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        states_trimmed = np.zeros((self.batch_size, self.num_agents, max(self.state_sizes)), dtype=np.float32)
        actions_trimmed = np.zeros((self.batch_size, self.num_agents, max(self.action_sizes)), dtype=np.float32)
        next_states_trimmed = np.zeros((self.batch_size, self.num_agents, max(self.state_sizes)), dtype=np.float32)
        
        for i in range(self.num_agents):
            states_trimmed[:, i, :self.state_sizes[i]] = states[:, i, :self.state_sizes[i]]
            actions_trimmed[:, i, :self.action_sizes[i]] = actions[:, i, :self.action_sizes[i]]
            next_states_trimmed[:, i, :self.state_sizes[i]] = next_states[:, i, :self.state_sizes[i]]
        
        return states_trimmed, actions_trimmed, rewards, next_states_trimmed, dones
    
    def __len__(self):
        return self.size