import numpy as np
from pettingzoo.mpe import simple_spread_v3

class RelayRaceWrapper:
    def __init__(self, base_env, tag_distance=0.1):
        self.base_env = base_env
        self.tag_distance = tag_distance
        self._agents = None
        self.num_agents = None
        self.agent_active = None
        self.agent_reached_landmark = None
        self.agent_positions = None
        self.landmark_positions = None
        self.previous_distances = None
        
    def reset(self, seed=None, options=None):
        observations, infos = self.base_env.reset(seed=seed, options=options)

        if self._agents is None:
            self._agents = list(observations.keys())
            self.num_agents = len(self._agents)
            self.agent_active = np.zeros(self.num_agents, dtype=bool)
            self.agent_reached_landmark = np.zeros(self.num_agents, dtype=bool)

        self.agent_active.fill(False)
        self.agent_active[0] = True
        self.agent_reached_landmark.fill(False)

        self._update_positions()
        self.previous_distances = self._compute_distances_to_landmarks()
        
        return observations, infos
    
    def step(self, actions):
        if self.agent_positions is not None and self.landmark_positions is not None:
            prev_distances = self._compute_distances_to_landmarks()
        else:
            prev_distances = None
        observations, rewards, terminations, truncations, infos = self.base_env.step(actions)

        self._update_positions()
        self._update_landmark_reached()
        self._update_active_status()
        modified_rewards = self._compute_relay_rewards(rewards, prev_distances)
        
        return observations, modified_rewards, terminations, truncations, infos
    
    def _update_positions_from_observations(self, observations):
        self._update_positions()
        if self.agent_positions is None or np.all(self.agent_positions == 0):
            if observations and len(observations) > 0:
                first_agent = list(observations.keys())[0]
                obs = observations[first_agent]
                if len(obs) >= 4:
                    pass
    
    def _update_positions(self):
        try:
            env = self.base_env
            while hasattr(env, 'env') or hasattr(env, 'aec_env'):
                if hasattr(env, 'env'):
                    env = env.env
                elif hasattr(env, 'aec_env'):
                    env = env.aec_env
                else:
                    break
            if hasattr(env, 'world'):
                world = env.world
                self.agent_positions = np.array([[agent.state.p_pos[0], agent.state.p_pos[1]] 
                                                 for agent in world.agents])
                self.landmark_positions = np.array([[landmark.state.p_pos[0], landmark.state.p_pos[1]] 
                                                    for landmark in world.landmarks])
                return
        except Exception as e:
            pass

        if self.agent_positions is None:
            self.agent_positions = np.zeros((self.num_agents, 2))
            self.landmark_positions = np.zeros((self.num_agents, 2))
    
    def _compute_distances_to_landmarks(self):
        distances = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if i < len(self.agent_positions) and i < len(self.landmark_positions):
                dist = np.linalg.norm(self.agent_positions[i] - self.landmark_positions[i])
                distances[i] = dist
        return distances
    
    def _update_landmark_reached(self):
        distances = self._compute_distances_to_landmarks()
        for i in range(self.num_agents):
            if distances[i] < self.tag_distance:
                self.agent_reached_landmark[i] = True
    
    def _update_active_status(self):
        if not self.agent_active[0]:
            self.agent_active[0] = True

        for i in range(1, self.num_agents):
            if self.agent_reached_landmark[i-1]:
                self.agent_active[i] = True
    
    def _compute_relay_rewards(self, base_rewards, prev_distances=None):
        modified_rewards = {}
        current_distances = self._compute_distances_to_landmarks()
        if prev_distances is not None:
            distances_to_use = prev_distances
        elif self.previous_distances is not None:
            distances_to_use = self.previous_distances
        else:
            distances_to_use = None
        
        for idx, agent in enumerate(self._agents):
            if self.agent_active[idx]:
                if distances_to_use is not None:
                    progress = distances_to_use[idx] - current_distances[idx]
                    progress_reward = max(0, progress) * 10.0 
                else:
                    progress_reward = 0.0
                if self.agent_reached_landmark[idx]:
                    landmark_bonus = 10.0
                else:
                    landmark_bonus = 0.0
                modified_rewards[agent] = base_rewards.get(agent, 0.0) + progress_reward + landmark_bonus
            else:
                modified_rewards[agent] = 0.0
        self.previous_distances = current_distances.copy()
        
        return modified_rewards
    
    def __getattr__(self, name):
        if name == 'agents':
            if self._agents is not None:
                return self._agents
            try:
                return getattr(self.base_env, name)
            except AttributeError:
                return None
        return getattr(self.base_env, name)


def make_relay_race_env(max_cycles=25, continuous_actions=True, tag_distance=0.1, render_mode=None):
    base_env = simple_spread_v3.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        render_mode=render_mode
    )
    
    wrapped_env = RelayRaceWrapper(base_env, tag_distance=tag_distance)
    return wrapped_env

if __name__ == "__main__":
    env = make_relay_race_env(max_cycles=25, tag_distance=0.1)
    print(f"Number of agents: {len(env.base_env.agents)}")
    
    observations, infos = env.reset()
    print(f"\nInitial observations: {len(observations)} agents")
    print(f"Initial active status: {env.agent_active}")
    for step in range(5):
        actions = {agent: env.base_env.action_space(agent).sample() 
                  for agent in env.base_env.agents}
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        print(f"  Active agents: {env.agent_active}")
        print(f"  Reached landmarks: {env.agent_reached_landmark}")
        print(f"  Rewards: {rewards}")
        
        if any(terminations.values()) or any(truncations.values()):
            break
    
    env.close()

