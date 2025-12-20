"""
FIXED Training script - addresses stagnant learning issue

File: experiments/train_baseline.py

Usage:
    python experiments/train_baseline.py --n_episodes 2000 --seed 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from datetime import datetime

from envs.mpe_wrapper import MPEWrapper
from algorithms.mappo import MAPPO


def collect_episode(env, mappo, max_steps=25):
    """
    Collect a single episode of data
    
    Returns:
        episode_data: Dict with trajectory data for each agent
        episode_reward: Total episode reward
    """
    obs, _, _ = env.reset()
    
    # Episode storage
    episode_data = {i: {
        'obs': [],
        'actions': [],
        'log_probs': [],
        'values': [],
        'rewards': [],
        'dones': []
    } for i in range(mappo.n_agents)}
    
    episode_reward = 0
    
    for step in range(max_steps):
        # Select actions
        actions, log_probs, values = mappo.select_actions(obs)
        
        # Store data (CRITICAL: convert to tensors properly)
        for i in range(mappo.n_agents):
            episode_data[i]['obs'].append(torch.FloatTensor(obs[i]))
            episode_data[i]['actions'].append(torch.LongTensor([actions[i]]))
            episode_data[i]['log_probs'].append(torch.FloatTensor([log_probs[i]]))
            episode_data[i]['values'].append(values[i])  # Keep as float
        
        # Step environment
        obs, _, rewards, dones, _, _ = env.step(actions)
        
        # Store rewards and dones
        for i in range(mappo.n_agents):
            episode_data[i]['rewards'].append(float(rewards[i, 0]))
            episode_data[i]['dones'].append(float(dones[i]))
        
        episode_reward += float(rewards.sum())
        
        if dones.all():
            break
    
    return episode_data, episode_reward


def train_mappo(
    env_name='simple_spread',
    n_agents=3,
    n_episodes=2000,
    max_steps=25,
    hidden_dim=64,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    save_dir='./results',
    save_interval=200,
    log_interval=20,
    seed=1,
    eval_interval=100  # NEW: periodic evaluation
):
    """
    Train MAPPO baseline with proper data handling
    """
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Force deterministic behavior on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create environment
    env = MPEWrapper(env_name=env_name, n_agents=n_agents, max_cycles=max_steps)
    
    # Get dimensions
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create MAPPO
    mappo = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device
    )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir) / env_name / f'seed_{seed}' / timestamp
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MAPPO Baseline Training - FIXED VERSION")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Agents: {n_agents}")
    print(f"Episodes: {n_episodes}")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Save path: {save_path}")
    print("=" * 70 + "\n")
    
    # Training stats
    recent_rewards = []
    all_rewards = []
    best_avg_reward = float('-inf')
    episodes_without_improvement = 0
    
    print("Starting training...")
    print("Expected progress: -85 → -60 → -40 → -20 → -15")
    print("-" * 70 + "\n")
    
    # Training loop
    for episode in range(n_episodes):
        # Collect episode
        episode_data, episode_reward = collect_episode(env, mappo, max_steps)
        
        # Update policy
        mappo.update(episode_data)
        
        # Track rewards
        recent_rewards.append(episode_reward)
        all_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        avg_reward = np.mean(recent_rewards) if len(recent_rewards) >= 10 else episode_reward
        
        # Check for improvement
        if avg_reward > best_avg_reward and episode > 50:
            best_avg_reward = avg_reward
            episodes_without_improvement = 0
            mappo.save(save_path / 'best_model.pt')
        else:
            episodes_without_improvement += 1
        
        # Early stopping if stuck
        if episodes_without_improvement > 500:
            print(f"\nWARNING: No improvement for 500 episodes. Early stopping.")
            print(f"Current avg: {avg_reward:.2f}, Best: {best_avg_reward:.2f}")
            break
        
        # Logging
        if (episode + 1) % log_interval == 0:
            # Show more stats for debugging
            recent_10 = recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards
            std_recent = np.std(recent_10)
            
            print(f"Ep {episode+1:4d} | "
                  f"R: {episode_reward:7.2f} | "
                  f"Avg100: {avg_reward:7.2f} | "
                  f"Best: {best_avg_reward:7.2f} | "
                  f"Std10: {std_recent:6.2f} | "
                  f"NoImprove: {episodes_without_improvement}")
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_policy(mappo, env, n_episodes=10)
            print(f"  → Eval (10 eps): {eval_reward:.2f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            mappo.save(save_path / f'checkpoint_ep{episode+1}.pt')
            print(f"  → Checkpoint saved (avg: {avg_reward:.2f}, best: {best_avg_reward:.2f})")
    
    # Save final model
    mappo.save(save_path / 'final_model.pt')
    
    # Save training stats
    np.save(save_path / 'rewards.npy', np.array(all_rewards))
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final avg (last 100): {np.mean(recent_rewards):.2f}")
    print(f"Best avg: {best_avg_reward:.2f}")
    print(f"Total episodes: {episode + 1}")
    print(f"Models saved to: {save_path}")
    
    # Final evaluation
    print("\nFinal evaluation (100 episodes)...")
    final_eval = evaluate_policy(mappo, env, n_episodes=100)
    print(f"Final evaluation reward: {final_eval:.2f}")
    
    # Performance assessment
    if final_eval > -20:
        print("\n✓✓ EXCELLENT: Strong baseline achieved!")
    elif final_eval > -30:
        print("\n✓ GOOD: Acceptable baseline")
    elif final_eval > -50:
        print("\n⚠ MARGINAL: Consider training longer")
    else:
        print("\n✗ POOR: Something may be wrong")
    
    print("=" * 70)
    
    return mappo, all_rewards, save_path


def evaluate_policy(mappo, env, n_episodes=10):
    """Quick evaluation during training"""
    mappo.eval_mode()
    
    eval_rewards = []
    for _ in range(n_episodes):
        obs, _, _ = env.reset()
        episode_reward = 0
        
        for _ in range(25):
            actions, _, _ = mappo.select_actions(obs, deterministic=True)
            obs, _, rewards, dones, _, _ = env.step(actions)
            episode_reward += rewards.sum()
            
            if dones.all():
                break
        
        eval_rewards.append(episode_reward)
    
    mappo.train_mode()
    return np.mean(eval_rewards)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MAPPO Baseline (FIXED)')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='simple_spread')
    parser.add_argument('--n_agents', type=int, default=3)
    
    # Training
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=1)
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=100)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Test with 2 agents first if 3 agents doesn't work
    if args.n_agents == 3:
        print("NOTE: If training doesn't improve, try --n_agents 2 first\n")
    
    mappo, rewards, save_path = train_mappo(
        env_name=args.env_name,
        n_agents=args.n_agents,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model: {save_path}/best_model.pt")
    print(f"✓ Run evaluation: python experiments/evaluate.py --model_path {save_path}/best_model.pt")