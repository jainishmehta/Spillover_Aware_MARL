import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maddpg import MADDPG, Actor, Critic
from spillover_analysis import estimate_influence_granger, prepare_trajectories, load_trajectory_data


class SRMAPG(MADDPG):
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64),
                 actor_lr=1e-3, critic_lr=2e-3, gamma=0.95, tau=0.01,
                 action_low=None, action_high=None, reward_scale=1.0, weight_decay=1e-4,
                 lambda_inf=0.1, lambda_kl=0.01, spsa_epsilon=0.01, 
                 influence_update_interval=1000, use_influence_regularization=True,
                 freeze_influence_matrix=False, influence_update_alpha=0.1):
        super().__init__(state_sizes, action_sizes, hidden_sizes,
                        actor_lr, critic_lr, gamma, tau,
                        action_low, action_high, reward_scale, weight_decay)
        
        self.lambda_inf = lambda_inf
        self.lambda_kl = lambda_kl
        self.spsa_epsilon = spsa_epsilon
        self.influence_update_interval = influence_update_interval
        self.use_influence_regularization = use_influence_regularization
        self.freeze_influence_matrix = freeze_influence_matrix
        self.influence_update_alpha = influence_update_alpha

        self.S_operator = np.ones((self.num_agents, self.num_agents)) / (self.num_agents * self.num_agents)
        self.influence_update_step = 0
        self.initial_matrix_sum = None

        self.recent_trajectories = []
        self.recent_states = []
        self.recent_actions = []
        self.trajectory_buffer_size = 1000

        self.jacobian_cache = None
        self.jacobian_cache_step = -1
        self.jacobian_update_interval = 100
        self.use_fast_jacobian = freeze_influence_matrix
    
    def update_influence_matrix(self, trajectories_data=None, alpha=0.2, max_lag=5):
        if self.freeze_influence_matrix:
            return
        
        try:
            if trajectories_data is None:
                if len(self.recent_trajectories) < 50:
                    return
                T = len(self.recent_trajectories)
                trajectories = np.zeros((T, self.num_agents))
                
                for t, traj in enumerate(self.recent_trajectories):
                    for agent_idx in range(self.num_agents):
                        if 'policy_norm' in traj:
                            trajectories[t, agent_idx] = traj['policy_norm'][agent_idx]
                        elif 'agent_value' in traj:
                            trajectories[t, agent_idx] = traj['agent_value'][agent_idx]
                        else:
                            if agent_idx < len(traj.get('actions', [])):
                                trajectories[t, agent_idx] = np.linalg.norm(traj['actions'][agent_idx])
            else:
                trajectories, _ = prepare_trajectories(trajectories_data, metric='policy_params', max_lag=max_lag)

            A, p_values, _ = estimate_influence_granger(
                trajectories,
                max_lag=max_lag,
                alpha=alpha,
                make_stationary_if_needed=False,
                quiet=True
            )

            A_float = A.astype(float)

            if np.sum(A_float) > 0:
                A_normalized = A_float / (np.sum(A_float) + 1e-10)
            else:
                if self.influence_update_step % 100 == 0:
                    print(f"Warning: Granger causality found no relationships. Keeping current influence matrix.")
                return

            old_sum = np.sum(self.S_operator)
            self.S_operator = (1 - self.influence_update_alpha) * self.S_operator + \
                              self.influence_update_alpha * A_normalized

            new_sum = np.sum(self.S_operator)
            if new_sum < 1e-10 and old_sum > 1e-10:
                print(f"Warning: Influence matrix sum dropped too low ({new_sum:.2e}). Reverting update.")
                self.S_operator = 0.9 * self.S_operator + 0.1 * A_normalized
                if np.sum(self.S_operator) < 1e-10:
                    return

            if np.sum(self.S_operator) > 0:
                self.S_operator = self.S_operator / np.sum(self.S_operator)
            
            self.influence_update_step += 1
            
        except Exception as e:
            if self.influence_update_step % 100 == 0:
                print(f"Warning: Could not update influence matrix: {e}")
    
    def estimate_jacobian_spsa(self, states, actions, use_fast=False):
        if use_fast and self.freeze_influence_matrix:
            return np.abs(self.S_operator) * 0.1
        
        batch_size = states.shape[0]
        jacobian = np.zeros((self.num_agents, self.num_agents))

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        
        states_for_actors = []
        for i in range(self.num_agents):
            states_for_actors.append(states_tensor[:, i, :self.state_sizes[i]])
        states_flat = torch.cat(states_for_actors, dim=1)

        with torch.no_grad():
            actions_for_actors = []
            for i in range(self.num_agents):
                actions_for_actors.append(actions_tensor[:, i, :self.action_sizes[i]])
            actions_flat = torch.cat(actions_for_actors, dim=1)

            baseline_q = []
            for agent_idx in range(self.num_agents):
                q = self.critics[agent_idx](states_flat, actions_flat)
                baseline_q.append(q.mean().item())

            epsilon = self.spsa_epsilon
            
            for i in range(self.num_agents):
                perturbed_actions_for_actors = [a.clone() for a in actions_for_actors]
                perturbation = torch.randn_like(perturbed_actions_for_actors[i]) * epsilon
                perturbed_actions_for_actors[i] = perturbed_actions_for_actors[i] + perturbation
                perturbed_actions_for_actors[i] = torch.clamp(
                    perturbed_actions_for_actors[i],
                    torch.FloatTensor(self.action_low[i]),
                    torch.FloatTensor(self.action_high[i])
                )
                
                perturbed_actions_flat = torch.cat(perturbed_actions_for_actors, dim=1)

                q_perturbed = []
                for agent_idx in range(self.num_agents):
                    q = self.critics[agent_idx](states_flat, perturbed_actions_flat)
                    q_perturbed.append(q.mean().item())
                for j in range(self.num_agents):
                    jacobian[j, i] = (q_perturbed[j] - baseline_q[j]) / epsilon
        
        return jacobian
    
    def compute_kl_divergence(self, states, agent_idx):
        states_tensor = torch.FloatTensor(states)

        current_actions = self.actors[agent_idx](states_tensor)

        kl_penalty = torch.mean(torch.sum(current_actions ** 2, dim=1))
        
        return kl_penalty
    
    def learn(self, batch, agent_idx, global_step=0):
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]
        if len(self.recent_trajectories) >= self.trajectory_buffer_size:
            self.recent_trajectories.pop(0)

        with torch.no_grad():
            policy_norms = []
            for i in range(self.num_agents):
                params = list(self.actors[i].parameters())
                param_norm = sum(p.norm().item() for p in params)
                policy_norms.append(param_norm)

        if isinstance(actions, torch.Tensor):
            actions_np = actions.numpy()
            states_np = states.numpy()
        else:
            actions_np = actions
            states_np = states
        
        self.recent_trajectories.append({
            'policy_norm': policy_norms,
            'actions': [actions_np[:, i] for i in range(self.num_agents)],
            'states': states_np
        })

        if (global_step % self.influence_update_interval == 0 and 
            self.use_influence_regularization and 
            len(self.recent_trajectories) >= 50):
            self.update_influence_matrix()
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        states_for_actors = []
        next_states_for_actors = []
        for i in range(self.num_agents):
            states_for_actors.append(states[:, i, :self.state_sizes[i]])
            next_states_for_actors.append(next_states[:, i, :self.state_sizes[i]])

        states_flat = torch.cat(states_for_actors, dim=1)
        next_states_flat = torch.cat(next_states_for_actors, dim=1)
        actions_for_actors = []
        for i in range(self.num_agents):
            actions_for_actors.append(actions[:, i, :self.action_sizes[i]])
        actions_flat = torch.cat(actions_for_actors, dim=1)

        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                next_action = self.target_actors[i](next_states_for_actors[i])
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
                current_action = self.actors[i](states_for_actors[i])
            else:
                current_action = actions_for_actors[i].detach()
            current_actions.append(current_action)
        current_actions_flat = torch.cat(current_actions, dim=1)

        pg_loss = -self.critics[agent_idx](states_flat, current_actions_flat).mean()
        spillover_loss = torch.tensor(0.0)
        if self.use_influence_regularization and np.sum(self.S_operator) > 0:
            try:
                should_update_jacobian = (
                    self.jacobian_cache is None or
                    global_step - self.jacobian_cache_step >= self.jacobian_update_interval
                )
                
                if should_update_jacobian:
                    if isinstance(states, torch.Tensor):
                        states_np = states.numpy()
                        actions_np = actions.numpy()
                    else:
                        states_np = states
                        actions_np = actions

                    jacobian = self.estimate_jacobian_spsa(
                        states_np,
                        actions_np,
                        use_fast=self.use_fast_jacobian
                    )

                    self.jacobian_cache = jacobian
                    self.jacobian_cache_step = global_step
                else:
                    jacobian = self.jacobian_cache

                S_tensor = torch.FloatTensor(self.S_operator)
                J_tensor = torch.FloatTensor(np.abs(jacobian))
                spillover_loss = self.lambda_inf * torch.sum(S_tensor * J_tensor)
            except Exception as e:
                if global_step % 1000 == 0:
                    print(f"Warning: Could not compute spillover penalty: {e}")
                pass

        kl_loss = torch.tensor(0.0)
        if self.lambda_kl > 0:
            kl_loss = self.lambda_kl * self.compute_kl_divergence(
                states_for_actors[agent_idx],
                agent_idx
            )

        actor_loss = pg_loss + spillover_loss + kl_loss

        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
        self.actor_optimizers[agent_idx].step()
        
        return critic_loss.item(), actor_loss.item(), spillover_loss.item(), kl_loss.item()
    
    def set_influence_matrix(self, S_operator, freeze=False):
        if S_operator.shape == (self.num_agents, self.num_agents):
            self.S_operator = np.array(S_operator)
            self.initial_matrix_sum = np.sum(self.S_operator)

            if np.sum(self.S_operator) > 0:
                self.S_operator = self.S_operator / np.sum(self.S_operator)
            
            if freeze:
                self.freeze_influence_matrix = True
                print(f"Influence matrix set and FROZEN (sum={self.initial_matrix_sum:.4f})")
            else:
                print(f"Influence matrix set (sum={self.initial_matrix_sum:.4f}), will update during training")
        else:
            raise ValueError(f"Influence matrix shape {S_operator.shape} doesn't match num_agents {self.num_agents}")
    
    def get_influence_matrix(self):
        return self.S_operator.copy()


if __name__ == "__main__":
    state_sizes = [18, 18, 18]  # Example: 3 agents with 18-dim state each
    action_sizes = [5, 5, 5]    # Example: 3 agents with 5-dim action each
    
    srmapg = SRMAPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        lambda_inf=0.1,
        lambda_kl=0.01,
        spsa_epsilon=0.01,
        influence_update_interval=1000
    )
    
    print("SR-MAPG initialized successfully!")
    print(f"Influence matrix shape: {srmapg.S_operator.shape}")
    print(f"Influence matrix:\n{srmapg.S_operator}")

