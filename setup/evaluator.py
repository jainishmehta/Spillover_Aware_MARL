import numpy as np


def _get_base_env(env):
    base_env = env
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    elif hasattr(env, 'base_env'):
        base_env = env.base_env
        while hasattr(base_env, 'env') or hasattr(base_env, 'aec_env'):
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            elif hasattr(base_env, 'aec_env'):
                base_env = base_env.aec_env
            else:
                break
    return base_env


def check_success_simple_spread(env):
    try:
        base_env = _get_base_env(env)
        if not hasattr(base_env, 'world'):
            return False
            
        agent_positions = np.array([agent.state.p_pos for agent in base_env.world.agents])
        landmark_positions = np.array([landmark.state.p_pos for landmark in base_env.world.landmarks])

        threshold = 0.6
        landmarks_covered = []
        for landmark_pos in landmark_positions:
            distances = np.linalg.norm(agent_positions - landmark_pos, axis=1)
            is_covered = np.any(distances < threshold)
            landmarks_covered.append(is_covered)
        return all(landmarks_covered)
        
    except (AttributeError, IndexError, TypeError):
        return False


def check_success_simple_spread_relay(env):
    try:
        if hasattr(env, 'agent_reached_landmark'):
            return np.all(env.agent_reached_landmark)

        return check_success_simple_spread(env)
    except (AttributeError, IndexError, TypeError):
        return False


def check_success_simple_adversary(env):
    """Check if good agents reached the target. Good agents are rewarded based on how close the closest 
    one of them is to the target landmark and negatively rewarded based on how close the adversary is to the target landmark."""
    try:
        base_env = _get_base_env(env)
        if not hasattr(base_env, 'world'):
            return False

        all_agents = base_env.world.agents
        agent_positions = np.array([agent.state.p_pos for agent in all_agents])
        landmarks = base_env.world.landmarks
        landmark_positions = np.array([landmark.state.p_pos for landmark in landmarks])

        agent_names = []
        if hasattr(env, 'agents'):
            agent_names = list(env.agents)
        elif hasattr(base_env, 'agents'):
            agent_names = list(base_env.agents)

        good_agent_indices = []
        if len(agent_names) == len(all_agents):
            for i, agent_name in enumerate(agent_names):
                if not ('adversary' in str(agent_name).lower()):
                    good_agent_indices.append(i)
        else:
            for i, agent in enumerate(all_agents):
                agent_name = getattr(agent, 'name', '')
                if not ('adversary' in str(agent_name).lower()):
                    good_agent_indices.append(i)

        if len(good_agent_indices) == 0:
            good_agent_indices = list(range(len(all_agents)))

        target_landmark_idx = 0
        for i, landmark in enumerate(landmarks):
            if hasattr(landmark, 'color'):
                color = landmark.color
                if len(color) >= 3 and color[1] > color[0] and color[1] > color[2]:
                    target_landmark_idx = i
                    break
        
        target_landmark_pos = landmark_positions[target_landmark_idx]
        good_agent_positions = agent_positions[good_agent_indices]

        adversary_indices = [i for i in range(len(all_agents)) if i not in good_agent_indices]

        distances_to_target = np.linalg.norm(good_agent_positions - target_landmark_pos, axis=1)
        closest_good_agent_distance = np.min(distances_to_target)
    
        adversary_distance_to_target = None
        if len(adversary_indices) > 0:
            adversary_positions = agent_positions[adversary_indices]
            adversary_distances = np.linalg.norm(adversary_positions - target_landmark_pos, axis=1)
            adversary_distance_to_target = np.min(adversary_distances)

        target_threshold = 0.6
        return closest_good_agent_distance < target_threshold
        
    except (AttributeError, IndexError, TypeError):
        return False


def check_success_simple_tag(env):
    try:
        base_env = _get_base_env(env)
        if not hasattr(base_env, 'world'):
            return False
        # return True if episode completed (agents survived)
        return True
        
    except (AttributeError, IndexError, TypeError):
        return False


def check_success(env, env_name=None):
    if env_name is None:
        env_str = str(type(env)).lower()
        if 'relay' in env_str or hasattr(env, 'agent_reached_landmark'):
            env_name = 'simple_spread_relay'
        elif 'adversary' in env_str:
            env_name = 'simple_adversary_v3'
        elif 'tag' in env_str:
            env_name = 'simple_tag_v3'
        elif 'spread' in env_str:
            env_name = 'simple_spread_v3'
        else:
            return check_success_simple_spread(env)

    if env_name == 'simple_spread_v3':
        return check_success_simple_spread(env)
    elif env_name == 'simple_spread_relay':
        return check_success_simple_spread_relay(env)
    elif env_name == 'simple_adversary_v3':
        return check_success_simple_adversary(env)
    elif env_name == 'simple_tag_v3':
        return check_success_simple_tag(env)
    else:
        return check_success_simple_spread(env)


def evaluate(env, maddpg, agents, logger, global_step, num_episodes=10, env_name=None):
    """
    Args:
        env: The environment to evaluate on
        maddpg: The MADDPG agent (or compatible agent with act() method)
        agents: List of agent names
        logger: Logger object for logging metrics
        global_step: Current training step
        num_episodes: Number of episodes to evaluate
        env_name: Optional environment name for success checking. If None, will auto-detect.
    
    Returns:
        tuple: (avg_rewards, success_rate)
    """
    eval_episode_rewards = []
    successful_episodes = 0
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        episode_rewards = np.zeros(num_agents)
        observations, _ = env.reset()
        
        done = False
        step = 0
        episode_success = False
        
        while not done:
            states = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            actions = maddpg.act(states, noise_scale=0.0)
            actions_dict = {agent: action for agent, action in zip(agents, actions)}
            next_observations, rewards, terminations, truncations, _ = env.step(actions_dict)
            rewards_array = np.array([rewards[agent] for agent in agents])
            episode_rewards += rewards_array
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            if done or step >= 24: 
                # Check success for all environments using general function
                episode_success = check_success(env, env_name=env_name)
            
            observations = next_observations
            step += 1
        
        eval_episode_rewards.append(episode_rewards)
        if episode_success:
            successful_episodes += 1
    avg_rewards = np.mean(eval_episode_rewards, axis=0)
    success_rate = (successful_episodes / num_episodes) * 100.0

    for i, agent in enumerate(agents):
        logger.log_scalar(f'eval/{agent}_reward', avg_rewards[i], global_step)

    logger.log_scalar('eval/total_reward', np.sum(avg_rewards), global_step)
    logger.log_scalar('eval/success_rate', success_rate, global_step)

    print(f"\n[Step {global_step}]")
    for i, agent in enumerate(agents):
        print(f"  {agent}: {avg_rewards[i]:.2f}")
    print(f"  Success Rate: {success_rate:.1f}% ({successful_episodes}/{num_episodes})")
    
    return avg_rewards, success_rate