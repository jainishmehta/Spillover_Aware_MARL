import numpy as np


def check_success_simple_spread(env):
    try:
        if hasattr(env, 'unwrapped'):
            base_env = env.unwrapped
        else:
            base_env = env
        agent_positions = np.array([agent.state.p_pos for agent in base_env.world.agents])
        landmark_positions = np.array([landmark.state.p_pos for landmark in base_env.world.landmarks])
        
        # Distance threshold for "covering" a landmark
        threshold=0.6
        landmarks_covered = []
        for landmark_pos in landmark_positions:
            distances = np.linalg.norm(agent_positions - landmark_pos, axis=1)
            is_covered = np.any(distances < threshold)
            landmarks_covered.append(is_covered)
        return all(landmarks_covered)
        
    except (AttributeError, IndexError):
        return False


def evaluate(env, maddpg, agents, logger, global_step, num_episodes=10):
    """
    Args:
        env: Environment to evaluate on
        maddpg: MADDPG agent
        agents: List of agent names
        logger: Logger instance
        global_step: Current training step
        num_episodes: Number of episodes to evaluate
    
    Returns:
        avg_rewards: Average reward per agent
        success_rate: Percentage of successful episodes
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
                episode_success = check_success_simple_spread(env)
            
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

    print(f"\n[Step {global_step}] Evaluation ({num_episodes} episodes):")
    for i, agent in enumerate(agents):
        print(f"  {agent}: {avg_rewards[i]:.2f}")
    print(f"  Total: {np.sum(avg_rewards):.2f}")
    print(f"  Success Rate: {success_rate:.1f}% ({successful_episodes}/{num_episodes})")
    
    return avg_rewards, success_rate