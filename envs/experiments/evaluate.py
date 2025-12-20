"""
Evaluation script for trained MAPPO models

File: experiments/evaluate.py

Usage:
    python experiments/evaluate.py --model_path results/simple_spread/seed_1/best_model.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from envs.mpe_wrapper import MPEWrapper
from algorithms.mappo import MAPPO


def evaluate_policy(
    model_path,
    env_name='simple_spread',
    n_agents=3,
    n_episodes=100,
    render=False,
    seed=999
):
    """
    Evaluate a trained MAPPO policy
    
    Args:
        model_path: Path to saved model
        env_name: Environment name
        n_agents: Number of agents
        n_episodes: Number of evaluation episodes
        render: Render environment
        seed: Random seed
        
    Returns:
        avg_reward, std_reward, all_rewards
    """
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = MPEWrapper(env_name=env_name, n_agents=n_agents)
    
    # Get dimensions
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create MAPPO
    mappo = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Load model
    mappo.load(model_path)
    mappo.eval_mode()
    
    print("=" * 70)
    print("MAPPO Policy Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Agents: {n_agents}")
    print(f"Episodes: {n_episodes}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _, _ = env.reset()
        episode_reward = 0
        step = 0
        
        for step in range(25):
            # Deterministic actions
            actions, _, _ = mappo.select_actions(obs, deterministic=True)
            
            # Step
            obs, _, rewards, dones, _, _ = env.step(actions)
            episode_reward += rewards.sum()
            
            if render:
                env.render()
            
            if dones.all():
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if (episode + 1) % 20 == 0:
            print(f"Evaluated {episode + 1}/{n_episodes} episodes...")
    
    env.close()
    
    # Compute statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    median_reward = np.median(episode_rewards)
    
    avg_length = np.mean(episode_lengths)
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Average Reward:  {avg_reward:7.2f} ± {std_reward:6.2f}")
    print(f"Median Reward:   {median_reward:7.2f}")
    print(f"Min Reward:      {min_reward:7.2f}")
    print(f"Max Reward:      {max_reward:7.2f}")
    print(f"Average Length:  {avg_length:7.2f}")
    print("=" * 70)
    
    # Success criteria for simple_spread
    if env_name == 'simple_spread':
        print("\nPerformance Assessment:")
        if avg_reward > -15:
            print("  ✓✓✓ EXCELLENT: Reward > -15 (optimal performance)")
        elif avg_reward > -20:
            print("  ✓✓ GREAT: Reward > -20 (strong baseline)")
        elif avg_reward > -30:
            print("  ✓ GOOD: Reward > -30 (acceptable baseline)")
        else:
            print("  ⚠ NEEDS IMPROVEMENT: Train longer or tune hyperparameters")
        
        print(f"\nBaseline Quality:")
        print(f"  This policy is suitable for SR-MAPG comparison: {'YES ✓' if avg_reward > -30 else 'NO - train more'}")
    
    print("=" * 70)
    
    return avg_reward, std_reward, episode_rewards


def compare_models(model_paths, env_name='simple_spread', n_agents=3):
    """Compare multiple models"""
    
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    results = []
    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        avg, std, rewards = evaluate_policy(
            model_path,
            env_name=env_name,
            n_agents=n_agents,
            n_episodes=50  # Fewer episodes for comparison
        )
        results.append({
            'path': model_path,
            'avg': avg,
            'std': std
        })
    
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    for i, result in enumerate(results):
        print(f"{i+1}. {Path(result['path']).name:20s} | "
              f"Reward: {result['avg']:7.2f} ± {result['std']:6.2f}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MAPPO Policy')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--env_name', type=str, default='simple_spread',
                       help='Environment name')
    parser.add_argument('--n_agents', type=int, default=3,
                       help='Number of agents')
    parser.add_argument('--n_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--seed', type=int, default=999,
                       help='Random seed')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    avg, std, rewards = evaluate_policy(
        model_path=args.model_path,
        env_name=args.env_name,
        n_agents=args.n_agents,
        n_episodes=args.n_episodes,
        render=args.render,
        seed=args.seed
    )
    
    # Save results
    results_path = Path(args.model_path).parent / 'eval_results.npz'
    np.savez(
        results_path,
        avg_reward=avg,
        std_reward=std,
        all_rewards=rewards
    )
    print(f"\n✓ Results saved to: {results_path}")