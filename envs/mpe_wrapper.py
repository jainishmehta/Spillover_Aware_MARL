"""
MPE Environment Wrapper for MARL Research
Compatible with PettingZoo and custom algorithms

File: envs/mpe_wrapper.py
"""

import numpy as np
from gymnasium import spaces
from pettingzoo.mpe import simple_spread_v3


class MPEWrapper:
    """
    Wrapper for PettingZoo MPE environments
    Provides consistent interface for MARL algorithms
    """
    
    def __init__(self, env_name='simple_spread', n_agents=3, max_cycles=25, 
                 continuous_actions=False):
        """
        Args:
            env_name: Environment name ('simple_spread', 'simple_tag', etc.)
            n_agents: Number of agents
            max_cycles: Maximum steps per episode
            continuous_actions: Use continuous action space
        """
        self.env_name = env_name
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        
        # Create environment
        if env_name == 'simple_spread':
            self.env = simple_spread_v3.parallel_env(
                N=n_agents,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions
            )
        else:
            raise NotImplementedError(f"Environment {env_name} not implemented")
        
        # Initialize to get spaces
        obs_dict, _ = self.env.reset()
        self.agents = self.env.agents
        
        # Define spaces
        first_agent = self.agents[0]
        self.observation_space = [self.env.observation_space(first_agent)] * n_agents
        self.action_space = [self.env.action_space(first_agent)] * n_agents
        
        # For centralized critic (MAPPO)
        obs_dim = self.observation_space[0].shape[0]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, 
                      shape=(obs_dim * n_agents,), dtype=np.float32)
        ] * n_agents
        
        self.step_count = 0
    
    def reset(self, seed=None):
        """Reset environment"""
        if seed is not None:
            # PettingZoo doesn't use seed() anymore
            pass
        
        obs_dict, info_dict = self.env.reset(seed=seed)
        self.step_count = 0
        
        # Convert to array format
        obs = np.array([obs_dict[agent] for agent in self.agents])
        
        # Shared observation (concatenated local observations)
        share_obs = np.tile(obs.flatten(), (self.n_agents, 1))
        
        # Available actions (all available by default)
        available_actions = None
        if isinstance(self.action_space[0], spaces.Discrete):
            available_actions = np.ones((self.n_agents, self.action_space[0].n))
        
        return obs, share_obs, available_actions
    
    def step(self, actions):
        """
        Execute actions for all agents
        
        Args:
            actions: [n_agents] array of actions
            
        Returns:
            obs, share_obs, rewards, dones, infos, available_actions
        """
        self.step_count += 1
        
        # Convert to dict
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        
        # Step
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = \
            self.env.step(action_dict)
        
        # Convert to arrays
        obs = np.array([obs_dict[agent] for agent in self.agents])
        rewards = np.array([[reward_dict[agent]] for agent in self.agents])
        dones = np.array([done_dict[agent] or trunc_dict[agent] 
                         for agent in self.agents])
        infos = [info_dict[agent] for agent in self.agents]
        
        # Shared observation
        share_obs = np.tile(obs.flatten(), (self.n_agents, 1))
        
        # Available actions
        available_actions = None
        if isinstance(self.action_space[0], spaces.Discrete):
            available_actions = np.ones((self.n_agents, self.action_space[0].n))
        
        # Episode done
        if self.step_count >= self.max_cycles:
            dones = np.ones(self.n_agents, dtype=bool)
        
        return obs, share_obs, rewards, dones, infos, available_actions
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def render(self):
        """Render environment (if supported)"""
        try:
            return self.env.render()
        except:
            pass


def make_env(env_name='simple_spread', n_agents=3, seed=0):
    """Factory function to create environment"""
    def _init():
        env = MPEWrapper(env_name=env_name, n_agents=n_agents)
        return env
    return _init


if __name__ == "__main__":
    # Test the wrapper
    print("Testing MPE Wrapper...")
    print("=" * 60)
    
    env = MPEWrapper(env_name='simple_spread', n_agents=3)
    
    print(f"Environment: {env.env_name}")
    print(f"Agents: {env.n_agents}")
    print(f"Obs shape: {env.observation_space[0].shape}")
    print(f"Action space: {env.action_space[0]}")
    
    # Test reset
    obs, share_obs, avail = env.reset()
    print(f"\nReset successful")
    print(f"  obs: {obs.shape}")
    print(f"  share_obs: {share_obs.shape}")
    
    # Test step
    actions = np.random.randint(0, env.action_space[0].n, size=env.n_agents)
    obs, share_obs, rewards, dones, infos, avail = env.step(actions)
    print(f"\nStep successful")
    print(f"  rewards: {rewards.flatten()}")
    
    env.close()
    print("\n" + "=" * 60)
    print("✓ All tests passed!")