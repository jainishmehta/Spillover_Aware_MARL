"""
Environment Utilities for PettingZoo MPE
"""
import numpy as np
from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3


ENV_MAP = {
    'simple_spread_v3': simple_spread_v3,
    'simple_tag_v3': simple_tag_v3,
    'simple_adversary_v3': simple_adversary_v3,
}


def create_env(env_name, max_steps=25):
    """Create a PettingZoo environment"""
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}")
    
    env = ENV_MAP[env_name].parallel_env(
        max_cycles=max_steps,
        continuous_actions=True,
        render_mode=None
    )
    
    return env


def get_env_info(env_name, max_steps=25):
    """Get environment information without creating persistent env"""
    env = create_env(env_name, max_steps)
    env.reset()
    
    agents = env.agents
    num_agents = len(agents)
    
    # Get observation and action spaces
    state_sizes = []
    action_sizes = []
    action_low = []
    action_high = []
    
    for agent in agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        
        state_sizes.append(obs_space.shape[0])
        action_sizes.append(act_space.shape[0])
        action_low.append(act_space.low)
        action_high.append(act_space.high)
    
    env.close()
    
    return agents, num_agents, action_sizes, action_low, action_high, state_sizes