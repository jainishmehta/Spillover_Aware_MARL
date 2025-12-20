# ============================================================================
# FIXED COMPLETE MARL SETUP: Multi-Agent Particle Environment + MADDPG
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pettingzoo.mpe import simple_spread_v3

# ============================================================================
# PART 1: NEURAL NETWORKS
# ============================================================================

class Actor(nn.Module):
    """Policy network - maps observations to actions"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # FIX: Output [0, 1] instead of [-1, 1]
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """Value network - evaluates state-action pairs"""
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=128):
        super(Critic, self).__init__()
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents
        
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.network(x)


# ============================================================================
# PART 2: REPLAY BUFFER (FIXED)
# ============================================================================

class ReplayBuffer:
    """Store and sample experiences for training"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """experience = (states, actions, rewards, next_states, dones)"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # FIX: Convert to numpy arrays first, then to tensors
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PART 3: MADDPG AGENT
# ============================================================================

class MADDPGAgent:
    """Single agent in the MADDPG framework"""
    def __init__(self, agent_id, state_dim, action_dim, n_agents,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim, n_agents)
        self.critic_target = Critic(state_dim, action_dim, n_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def act(self, state, noise=0.1):
        """Select action with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        # Add exploration noise (actions already in [0, 1])
        if noise > 0:
            action += noise * np.random.randn(*action.shape)
            action = np.clip(action, 0, 1)  # FIX: Clip to [0, 1]
        
        return action
    
    def update(self, all_agents, batch, agent_idx):
        """Update actor and critic networks"""
        states, actions, rewards, next_states, dones = batch
        
        # Extract this agent's data
        agent_rewards = rewards[:, agent_idx].unsqueeze(1)
        agent_dones = dones[:, agent_idx].unsqueeze(1)
        
        # ---- Update Critic ----
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(all_agents):
                next_state = next_states[:, i, :]
                next_action = agent.actor_target(next_state)
                next_actions.append(next_action)
            next_actions = torch.cat(next_actions, dim=1)
            
            next_states_flat = next_states.reshape(next_states.shape[0], -1)
            target_q = self.critic_target(next_states_flat, next_actions)
            target_q = agent_rewards + self.gamma * (1 - agent_dones) * target_q
        
        states_flat = states.reshape(states.shape[0], -1)
        actions_flat = actions.reshape(actions.shape[0], -1)
        current_q = self.critic(states_flat, actions_flat)
        
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # ---- Update Actor ----
        current_actions = []
        for i, agent in enumerate(all_agents):
            if i == agent_idx:
                state = states[:, i, :]
                action = self.actor(state)
            else:
                state = states[:, i, :]
                action = agent.actor(state).detach()
            current_actions.append(action)
        current_actions = torch.cat(current_actions, dim=1)
        
        actor_loss = -self.critic(states_flat, current_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # ---- Soft update target networks ----
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        """Slowly update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )


# ============================================================================
# PART 4: MADDPG TRAINER (FIXED)
# ============================================================================

class MADDPGTrainer:
    """Coordinates training of all agents"""
    def __init__(self, env_name='simple_spread_v3', n_agents=3, continuous_actions=True):
        self.env = simple_spread_v3.parallel_env(
            N=n_agents, 
            max_cycles=25,
            continuous_actions=continuous_actions
        )
        self.env.reset()
        
        sample_agent = self.env.agents[0]
        self.state_dim = self.env.observation_space(sample_agent).shape[0]
        
        action_space = self.env.action_space(sample_agent)
        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
            self.action_dim = action_space.shape[0]
        else:
            self.action_dim = 5
        
        self.n_agents = len(self.env.agents)
        self.continuous_actions = continuous_actions
        
        print(f"Environment initialized:")
        print(f"  - Number of agents: {self.n_agents}")
        print(f"  - State dimension: {self.state_dim}")
        print(f"  - Action dimension: {self.action_dim}")
        print(f"  - Continuous actions: {self.continuous_actions}")
        
        self.agents = [
            MADDPGAgent(i, self.state_dim, self.action_dim, self.n_agents)
            for i in range(self.n_agents)
        ]
        
        self.buffer = ReplayBuffer(capacity=100000)
        self.episode_rewards = []
        
    def collect_experience(self, n_episodes=1):
        """Run episodes and collect experiences"""
        for _ in range(n_episodes):
            observations, infos = self.env.reset()
            episode_reward = 0
            
            for step in range(25):
                # Get actions from all agents
                actions = {}
                for agent_name in self.env.agents:
                    agent_idx = self.env.agents.index(agent_name)
                    state = observations[agent_name]
                    action = self.agents[agent_idx].act(state, noise=0.1)
                    actions[agent_name] = action
                
                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # FIX: Properly handle agent lists that might change
                current_agents = list(observations.keys())
                
                # Store experience only if we have valid data for all agents
                if len(current_agents) == self.n_agents:
                    states = np.array([observations[a] for a in current_agents])
                    actions_array = np.array([actions[a] for a in current_agents])
                    rewards_array = np.array([rewards.get(a, 0.0) for a in current_agents])
                    next_states = np.array([next_observations.get(a, np.zeros(self.state_dim)) for a in current_agents])
                    dones = np.array([terminations.get(a, True) or truncations.get(a, True) for a in current_agents])
                    
                    self.buffer.push((states, actions_array, rewards_array, next_states, dones))
                    episode_reward += sum(rewards.values())
                
                observations = next_observations
                
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            self.episode_rewards.append(episode_reward)
    
    def train_step(self, batch_size=64):
        """Update all agents"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = self.buffer.sample(batch_size)
        
        losses = []
        for i, agent in enumerate(self.agents):
            critic_loss, actor_loss = agent.update(self.agents, batch, i)
            losses.append((critic_loss, actor_loss))
        
        return losses
    
    def train(self, n_episodes=1000, batch_size=64, update_every=100):
        """Main training loop"""
        print("\nStarting training...")
        
        for episode in range(n_episodes):
            # Collect experience
            self.collect_experience(n_episodes=1)
            
            # Train agents
            if len(self.buffer) >= batch_size:
                for _ in range(10):
                    self.train_step(batch_size)
            
            # Logging
            if (episode + 1) % update_every == 0:
                avg_reward = np.mean(self.episode_rewards[-update_every:])
                print(f"Episode {episode + 1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Buffer: {len(self.buffer)}")
    
    def evaluate(self, n_episodes=10):
        """Evaluate trained agents"""
        success_count = 0
        eval_rewards = []
        
        for _ in range(n_episodes):
            observations, infos = self.env.reset()
            episode_reward = 0
            
            for step in range(25):
                actions = {}
                for agent_name in self.env.agents:
                    agent_idx = self.env.agents.index(agent_name)
                    state = observations[agent_name]
                    action = self.agents[agent_idx].act(state, noise=0)
                    actions[agent_name] = action
                
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                episode_reward += sum(rewards.values())
                observations = next_observations
                
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            eval_rewards.append(episode_reward)
            if episode_reward > -50:
                success_count += 1
        
        success_rate = success_count / n_episodes
        avg_reward = np.mean(eval_rewards)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Success Rate: {success_rate * 100:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        
        return success_rate, avg_reward


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize trainer
    trainer = MADDPGTrainer(n_agents=3, continuous_actions=True)
    
    # Train agents
    trainer.train(n_episodes=1000, batch_size=64, update_every=100)
    
    # Evaluate
    success_rate, avg_reward = trainer.evaluate(n_episodes=20)
    
    # Save models
    for i, agent in enumerate(trainer.agents):
        torch.save(agent.actor.state_dict(), f'agent_{i}_actor.pth')
        torch.save(agent.critic.state_dict(), f'agent_{i}_critic.pth')
    
    print("\nTraining complete! Models saved.")