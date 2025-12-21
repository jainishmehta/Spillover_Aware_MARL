import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64)):
        super(Actor, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            linear = nn.Linear(input_size, hidden_size)
            # Use smaller initialization to prevent largeparameter growth
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(0.5))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            input_size = hidden_size

        output_layer = nn.Linear(input_size, action_size)
        nn.init.orthogonal_(output_layer.weight, gain=0.01)
        nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, total_state_size, total_action_size, hidden_sizes=(64, 64)):
        super(Critic, self).__init__()
        
        layers = []
        input_size = total_state_size + total_action_size
        
        for hidden_size in hidden_sizes:
            linear = nn.Linear(input_size, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(0.5))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            input_size = hidden_size

        output_layer = nn.Linear(input_size, 1)
        nn.init.orthogonal_(output_layer.weight, gain=0.5)
        nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.network(x)


class MADDPG:
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64),
                 actor_lr=1e-3, critic_lr=2e-3, gamma=0.95, tau=0.01,
                 action_low=None, action_high=None, reward_scale=1.0, weight_decay=1e-4):
        self.num_agents = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.weight_decay = weight_decay

        if action_low is None:
            self.action_low = [np.array([-1.0] * size) for size in action_sizes]
        else:
            self.action_low = action_low
            
        if action_high is None:
            self.action_high = [np.array([1.0] * size) for size in action_sizes]
        else:
            self.action_high = action_high

        total_state_size = sum(state_sizes)
        total_action_size = sum(action_sizes)
        
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(self.num_agents):
            actor = Actor(state_sizes[i], action_sizes[i], hidden_sizes)
            target_actor = Actor(state_sizes[i], action_sizes[i], hidden_sizes)
            target_actor.load_state_dict(actor.state_dict())
            
            critic = Critic(total_state_size, total_action_size, hidden_sizes)
            target_critic = Critic(total_state_size, total_action_size, hidden_sizes)
            target_critic.load_state_dict(critic.state_dict())
            
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=self.weight_decay)
            critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=self.weight_decay)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
    
    def act(self, states, noise_scale=0.0):
        actions = []
        
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_raw = self.actors[i](state_tensor).squeeze(0).numpy()
            
            action_scaled = (action_raw + 1.0) / 2.0
            action_scaled = action_scaled * (self.action_high[i] - self.action_low[i]) + self.action_low[i]
            
            if noise_scale > 0:
                noise = np.random.randn(*action_scaled.shape) * noise_scale
                action_final = action_scaled + noise
                action_final = np.clip(action_final, self.action_low[i], self.action_high[i])
            else:
                action_final = action_scaled
            action_final = action_final.astype(np.float32)
            actions.append(action_final)
        
        return actions
    
    def learn(self, batch, agent_idx):
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        next_states_flat = next_states.view(batch_size, -1)

        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                next_action = self.target_actors[i](next_states[:, i])
                next_actions.append(next_action)
            next_actions_flat = torch.cat(next_actions, dim=1)

            q_next = self.target_critics[agent_idx](next_states_flat, next_actions_flat)
            scaled_rewards = rewards[:, agent_idx:agent_idx+1] * self.reward_scale
            q_target = scaled_rewards + self.gamma * (1 - dones[:, agent_idx:agent_idx+1]) * q_next
            q_target = torch.clamp(q_target, -100.0, 100.0)
        

        q_current = self.critics[agent_idx](states_flat, actions_flat)

        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
        self.critic_optimizers[agent_idx].step()

        current_actions = []
        for i in range(self.num_agents):
            if i == agent_idx:
                current_action = self.actors[i](states[:, i])
            else:
                current_action = actions[:, i].detach()
            current_actions.append(current_action)
        current_actions_flat = torch.cat(current_actions, dim=1)

        actor_loss = -self.critics[agent_idx](states_flat, current_actions_flat).mean()

        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
        self.actor_optimizers[agent_idx].step()
        
        return critic_loss.item(), actor_loss.item()
    
    def update_targets(self):
        for i in range(self.num_agents):
            self._soft_update(self.actors[i], self.target_actors[i])
            self._soft_update(self.critics[i], self.target_critics[i])
    
    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers]
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizers'][i])
    
    def get_value_estimates(self, states):
        """
        Args:
            states: List of state arrays, one per agent (each can be 1D or 2D)
            
        Returns:
            List of Q-value estimates, one per agent
        """
        with torch.no_grad():
            states_tensors = []
            for state in states:
                state_tensor = torch.FloatTensor(state)
                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                states_tensors.append(state_tensor)
    
            batch_size = states_tensors[0].shape[0]
            states_tensor = torch.stack(states_tensors, dim=1)
            states_flat = states_tensor.view(batch_size, -1)

            actions = []
            for i in range(self.num_agents):
                action = self.actors[i](states_tensor[:, i])
                actions.append(action)
            actions_flat = torch.cat(actions, dim=1)

            q_values = []
            for agent_idx in range(self.num_agents):
                q_value = self.critics[agent_idx](states_flat, actions_flat)
                q_values.append(q_value.cpu().numpy().flatten())
            
            return q_values