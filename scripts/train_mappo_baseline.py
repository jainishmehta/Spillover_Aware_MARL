"""
Training script for MAPPO baseline - FIXED VERSION
Adapted from on-policy library for your research

Save as: scripts/train_mappo_baseline.py

Usage:
    python scripts/train_mappo_baseline.py --env_name simple_spread --experiment_name test
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try to import from on-policy if available
    from onpolicy.config import get_config
    from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
    from onpolicy.runner.shared.mpe_runner import MPERunner
    USE_ONPOLICY = True
except ImportError:
    print("Warning: on-policy library not found. Using simple training loop.")
    USE_ONPOLICY = False

# Import your wrapper
from envs.mpe_wrapper import MPEWrapper, make_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train MAPPO baseline')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='simple_spread',
                       choices=['simple_spread', 'simple_tag', 'simple_adversary'])
    parser.add_argument('--num_agents', type=int, default=3)
    
    # Training
    parser.add_argument('--num_env_steps', type=int, default=2000000,
                       help='Total number of environment steps')
    parser.add_argument('--n_rollout_threads', type=int, default=1,  # Changed to 1 for simplicity
                       help='Number of parallel environments')
    parser.add_argument('--episode_length', type=int, default=25)
    parser.add_argument('--n_training_threads', type=int, default=1)
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default='mappo_baseline')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--cuda_deterministic', action='store_true', default=True)
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./results')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save model every N updates')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N updates')
    parser.add_argument('--use_eval', action='store_true', default=True)
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Evaluate every N updates')
    parser.add_argument('--eval_episodes', type=int, default=32)
    parser.add_argument('--n_eval_rollout_threads', type=int, default=1)
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--layer_N', type=int, default=1)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--critic_lr', type=float, default=7e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    
    # PPO
    parser.add_argument('--ppo_epoch', type=int, default=15)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--num_mini_batch', type=int, default=1)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_loss_coef', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=10.0)
    
    # Additional flags
    parser.add_argument('--use_centralized_V', action='store_true', default=True)
    parser.add_argument('--use_obs_instead_of_state', action='store_true', default=False)
    parser.add_argument('--use_ReLU', action='store_true', default=True)
    parser.add_argument('--use_feature_normalization', action='store_true', default=True)
    parser.add_argument('--use_orthogonal', action='store_true', default=True)
    parser.add_argument('--use_recurrent_policy', action='store_true', default=False)
    parser.add_argument('--use_gae', action='store_true', default=True)
    parser.add_argument('--use_clipped_value_loss', action='store_true', default=True)
    parser.add_argument('--use_max_grad_norm', action='store_true', default=True)
    parser.add_argument('--use_huber_loss', action='store_true', default=True)
    parser.add_argument('--use_value_active_masks', action='store_true', default=True)
    parser.add_argument('--use_policy_active_masks', action='store_true', default=True)
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False)
    
    # Additional parameters
    parser.add_argument('--gain', type=float, default=0.01)
    parser.add_argument('--opti_eps', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--recurrent_N', type=int, default=1)
    parser.add_argument('--data_chunk_length', type=int, default=10)
    parser.add_argument('--huber_delta', type=float, default=10.0)
    
    args = parser.parse_args()
    return args


def simple_train(config):
    """
    Simple training loop without on-policy library
    Use this if on-policy import fails
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    
    print("\n" + "=" * 60)
    print("Using SIMPLE training loop (on-policy not available)")
    print("=" * 60)
    
    # Create environment
    env = MPEWrapper(config.env_name, config.num_agents)
    
    # Get dimensions
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # Simple actor-critic network
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        
        def forward(self, obs):
            return self.actor(obs), self.critic(obs)
    
    # Create agents
    agents = [ActorCritic() for _ in range(config.num_agents)]
    optimizers = [optim.Adam(agent.parameters(), lr=config.lr) 
                  for agent in agents]
    
    # Training loop
    num_episodes = config.num_env_steps // config.episode_length
    recent_rewards = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    print(f"Environment: {config.env_name}, Agents: {config.num_agents}")
    print("=" * 60 + "\n")
    
    for episode in range(num_episodes):
        obs, _, _ = env.reset()
        episode_data = {i: {'obs': [], 'actions': [], 'log_probs': [], 
                           'values': [], 'rewards': []} 
                       for i in range(config.num_agents)}
        episode_reward = 0
        
        # Collect episode
        for step in range(config.episode_length):
            obs_tensor = torch.FloatTensor(obs)
            
            # Select actions
            actions_list = []
            for i, agent in enumerate(agents):
                logits, value = agent(obs_tensor[i])
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                episode_data[i]['obs'].append(obs_tensor[i])
                episode_data[i]['actions'].append(action)
                episode_data[i]['log_probs'].append(log_prob)
                episode_data[i]['values'].append(value)
                actions_list.append(action.item())
            
            # Step environment
            obs, _, rewards, dones, _, _ = env.step(np.array(actions_list))
            
            for i in range(config.num_agents):
                episode_data[i]['rewards'].append(rewards[i, 0])
            
            episode_reward += rewards.sum()
            
            if dones.all():
                break
        
        # Update agents
        for i in range(config.num_agents):
            obs_batch = torch.stack(episode_data[i]['obs'])
            actions_batch = torch.stack(episode_data[i]['actions'])
            old_log_probs = torch.stack(episode_data[i]['log_probs'])
            values_batch = torch.stack(episode_data[i]['values']).squeeze()
            rewards_list = episode_data[i]['rewards']
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards_list):
                R = r + config.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            
            # Advantages
            advantages = returns - values_batch.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(config.ppo_epoch):
                logits, new_values = agents[i](obs_batch)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_batch)
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-config.clip_param, 
                                   1+config.clip_param) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values.squeeze(), returns)
                entropy_loss = -entropy.mean()
                
                loss = (policy_loss + 
                       config.value_loss_coef * value_loss + 
                       config.entropy_coef * entropy_loss)
                
                optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents[i].parameters(), 
                                       config.max_grad_norm)
                optimizers[i].step()
        
        # Logging
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f}")
        
        # Save
        if (episode + 1) % config.save_interval == 0:
            save_dir = Path(config.log_dir) / config.env_name / config.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                f'agent_{i}': agent.state_dict() 
                for i, agent in enumerate(agents)
            }, save_dir / f'checkpoint_{episode+1}.pt')
    
    print(f"\nTraining complete! Final avg reward: {np.mean(recent_rewards):.2f}")
    env.close()


def main():
    config = parse_args()
    
    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    
    # Create directories
    run_dir = Path(config.log_dir) / config.env_name / config.experiment_name / f'seed_{config.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Starting MAPPO Training")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
    print(f"Num agents: {config.num_agents}")
    print(f"Rollout threads: {config.n_rollout_threads}")
    print(f"Total steps: {config.num_env_steps}")
    print(f"Results dir: {run_dir}")
    print("=" * 60)
    
    if not USE_ONPOLICY:
        # Use simple training loop
        simple_train(config)
    else:
        # Use on-policy library (if available and working)
        from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
        
        # Make environments
        def make_train_env(rank):
            def init_env():
                env = MPEWrapper(config.env_name, config.num_agents)
                env.seed(config.seed + rank * 1000)
                return env
            return init_env
        
        if config.n_rollout_threads == 1:
            envs = DummyVecEnv([make_train_env(0)])
        else:
            envs = SubprocVecEnv([make_train_env(i) 
                                 for i in range(config.n_rollout_threads)])
        
        eval_envs = DummyVecEnv([make_train_env(0)]) if config.use_eval else None
        
        # Create runner
        try:
            from onpolicy.runner.shared.mpe_runner import MPERunner
            runner = MPERunner(
                config=config,
                env=envs,
                eval_env=eval_envs,
                num_agents=config.num_agents,
                run_dir=run_dir
            )
            runner.run()
            runner.save()
        except Exception as e:
            print(f"\nError with on-policy runner: {e}")
            print("Falling back to simple training...")
            envs.close()
            if eval_envs:
                eval_envs.close()
            simple_train(config)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Models saved to: {run_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()