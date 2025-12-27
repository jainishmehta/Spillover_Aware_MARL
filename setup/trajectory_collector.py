import numpy as np
import torch
import os
import pickle
from collections import defaultdict


class TrajectoryCollector:
    def __init__(self, num_agents, save_dir, collect_interval=1000):
        """
        Args:
            num_agents: Number of agents
            save_dir: Directory to save trajectory data
            collect_interval: How often to collect data (in training steps)
        """
        self.num_agents = num_agents
        self.save_dir = save_dir
        self.collect_interval = collect_interval

        self.timesteps = []
        self.policy_params = defaultdict(list)
        self.agent_values = defaultdict(list)
        self.episode_rewards = defaultdict(list)
        self.episode_timesteps = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def extract_policy_params(self, actor_network):
        """
        Args:
            actor_network: PyTorch actor network
            
        Returns:
            Flattened parameter vector as numpy array
        """
        params = []
        for param in actor_network.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def collect(self, timestep, maddpg, states=None, episode_rewards=None):
        if episode_rewards is not None:
            self.episode_timesteps.append(timestep)
            for agent_idx in range(self.num_agents):
                self.episode_rewards[agent_idx].append(episode_rewards[agent_idx])

        if timestep % self.collect_interval != 0:
            return
        
        self.timesteps.append(timestep)

        for agent_idx in range(self.num_agents):
            theta = self.extract_policy_params(maddpg.actors[agent_idx])
            self.policy_params[agent_idx].append(theta)
        if states is not None:
            q_values = maddpg.get_value_estimates(states)
            for agent_idx in range(self.num_agents):
                if len(q_values[agent_idx]) > 0:
                    self.agent_values[agent_idx].append(q_values[agent_idx][0])
                else:
                    self.agent_values[agent_idx].append(0.0)
    
    def save(self, filename="trajectory_data.pkl"):
        """
        Args:
            filename: Name of the file to save
        """
        trajectory_data = {
            'timesteps': np.array(self.timesteps),
            'policy_params': {agent_idx: np.array(params_list) 
                            for agent_idx, params_list in self.policy_params.items()},
            'agent_values': {agent_idx: np.array(values_list) 
                           for agent_idx, values_list in self.agent_values.items()},
            'episode_rewards': {agent_idx: np.array(rewards_list) 
                              for agent_idx, rewards_list in self.episode_rewards.items()},
            'episode_timesteps': np.array(self.episode_timesteps),
            'num_agents': self.num_agents,
            'collect_interval': self.collect_interval
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        print(f"\nTrajectory data saved to {filepath}")
        print(f"  - Collected {len(self.timesteps)} snapshots")
        print(f"  - Policy parameters shape: {[v.shape for v in trajectory_data['policy_params'].values()]}")
        print(f"  - Value estimates collected: {len(self.agent_values[0]) if 0 in self.agent_values else 0}")
        print(f"  - Episode rewards collected: {len(self.episode_rewards[0]) if 0 in self.episode_rewards else 0}")
    
    def get_trajectory_summary(self):
        """
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_snapshots': len(self.timesteps),
            'timestep_range': (min(self.timesteps), max(self.timesteps)) if self.timesteps else (0, 0),
            'policy_param_shapes': {agent_idx: [p.shape for p in params_list[:3]] 
                                  for agent_idx, params_list in self.policy_params.items()},
            'num_value_estimates': {agent_idx: len(values_list) 
                                   for agent_idx, values_list in self.agent_values.items()},
            'num_episode_rewards': {agent_idx: len(rewards_list) 
                                  for agent_idx, rewards_list in self.episode_rewards.items()}
        }
        return summary

