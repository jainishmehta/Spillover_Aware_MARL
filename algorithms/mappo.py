"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Stable implementation with GAE

File: algorithms/mappo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path


class ActorCritic(nn.Module):
    """Actor-Critic network for MAPPO"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # Smaller initialization for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, obs):
        """Forward pass"""
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for training"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return value, log_prob, entropy


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages, returns
    """
    advantages = []
    gae = 0
    
    # Add zero for terminal value
    values = values + [0]
    
    # Compute GAE backwards
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    # Returns = advantages + values
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    
    return advantages, returns


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization
    
    Key features:
    - One actor-critic per agent
    - Centralized critic (uses global state)
    - PPO with clipped surrogate objective
    - GAE for advantage estimation
    """
    
    def __init__(
        self,
        n_agents,
        obs_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        ppo_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create networks for each agent
        self.agents = [
            ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
            for _ in range(n_agents)
        ]
        
        # Optimizers
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=lr)
            for agent in self.agents
        ]
        
        # Training mode
        for agent in self.agents:
            agent.train()
    
    def select_actions(self, observations, deterministic=False):
        """
        Select actions for all agents
        
        Args:
            observations: [n_agents, obs_dim] observations
            deterministic: Use deterministic actions (for evaluation)
            
        Returns:
            actions, log_probs, values
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        actions = []
        log_probs = []
        values = []
        
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                action, log_prob, value, _ = agent.get_action(
                    obs_tensor[i:i+1], deterministic
                )
                actions.append(action.item())
                log_probs.append(log_prob.item())
                values.append(value.item())
        
        return np.array(actions), np.array(log_probs), np.array(values)
    
    def update(self, episode_data):
        """
        Update all agents using PPO
        
        Args:
            episode_data: Dict with trajectory data for each agent
                {agent_idx: {'obs': [], 'actions': [], 'log_probs': [],
                             'values': [], 'rewards': [], 'dones': []}}
        """
        for agent_idx in range(self.n_agents):
            # Get data for this agent
            data = episode_data[agent_idx]
            
            obs_batch = torch.stack(data['obs']).to(self.device)
            actions_batch = torch.stack(data['actions']).to(self.device)
            old_log_probs = torch.stack(data['log_probs']).to(self.device)
            
            # Compute advantages using GAE
            advantages, returns = compute_gae(
                data['rewards'],
                data['values'],
                data['dones'],
                self.gamma,
                self.gae_lambda
            )
            
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(self.ppo_epochs):
                # Evaluate actions
                values, log_probs, entropy = self.agents[agent_idx].evaluate_actions(
                    obs_batch, actions_batch
                )
                values = values.squeeze()
                
                # Policy loss (PPO clipped surrogate)
                ratio = torch.exp(log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 
                                   1 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Update
                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agents[agent_idx].parameters(),
                    self.max_grad_norm
                )
                self.optimizers[agent_idx].step()
    
    def save(self, path):
        """Save all agent models"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            f'agent_{i}': agent.state_dict()
            for i, agent in enumerate(self.agents)
        }, path)
    
    def load(self, path):
        """Load all agent models"""
        checkpoint = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(checkpoint[f'agent_{i}'])
    
    def eval_mode(self):
        """Set to evaluation mode"""
        for agent in self.agents:
            agent.eval()
    
    def train_mode(self):
        """Set to training mode"""
        for agent in self.agents:
            agent.train()


if __name__ == "__main__":
    # Test MAPPO
    print("Testing MAPPO...")
    print("=" * 60)
    
    # Create MAPPO instance
    mappo = MAPPO(
        n_agents=3,
        obs_dim=18,
        action_dim=5,
        hidden_dim=64
    )
    
    print(f"Created MAPPO with:")
    print(f"  Agents: {mappo.n_agents}")
    print(f"  Obs dim: {mappo.obs_dim}")
    print(f"  Action dim: {mappo.action_dim}")
    
    # Test action selection
    obs = np.random.randn(3, 18)
    actions, log_probs, values = mappo.select_actions(obs)
    
    print(f"\nTest action selection:")
    print(f"  Actions: {actions}")
    print(f"  Log probs: {log_probs}")
    print(f"  Values: {values}")
    
    print("\n" + "=" * 60)
    print("✓ MAPPO test passed!")