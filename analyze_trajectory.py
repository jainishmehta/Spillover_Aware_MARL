import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def load_trajectory_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def print_data_summary(data):
    print("\n" + "="*60)
    print("TRAJECTORY DATA SUMMARY")
    print("="*60)
    
    print(f"\nNumber of agents: {data['num_agents']}")
    print(f"Collection interval: {data['collect_interval']} steps")
    print(f"Total snapshots collected: {len(data['timesteps'])}")
    
    if len(data['timesteps']) > 0:
        print(f"Timestep range: {data['timesteps'][0]} to {data['timesteps'][-1]}")
        print(f"Total training steps covered: {data['timesteps'][-1] - data['timesteps'][0]}")
    
    print("\nPolicy Parameters (θ_i):")
    for agent_idx in range(data['num_agents']):
        if agent_idx in data['policy_params']:
            params = data['policy_params'][agent_idx]
            print(f"  Agent {agent_idx}: {params.shape} (snapshots × parameters)")
            print(f"    Parameter dimension: {params.shape[1] if len(params.shape) > 1 else len(params)}")
    
    print("\nAgent Values (V_i - Q-values):")
    for agent_idx in range(data['num_agents']):
        if agent_idx in data['agent_values']:
            values = data['agent_values'][agent_idx]
            print(f"  Agent {agent_idx}: {len(values)} value estimates")
            if len(values) > 0:
                print(f"    Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
                print(f"    Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
    
    print("\nEpisode Rewards:")
    for agent_idx in range(data['num_agents']):
        if agent_idx in data['episode_rewards']:
            rewards = data['episode_rewards'][agent_idx]
            print(f"  Agent {agent_idx}: {len(rewards)} episode rewards")
            if len(rewards) > 0:
                print(f"    Mean: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}")
                print(f"    Range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    
    print("\n" + "="*60)


def plot_trajectories(data, save_dir=None):
    num_agents = data['num_agents']
    timesteps = data['timesteps']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trajectory Data Analysis', fontsize=16)

    ax1 = axes[0, 0]
    for agent_idx in range(num_agents):
        if agent_idx in data['agent_values'] and len(data['agent_values'][agent_idx]) > 0:
            values = data['agent_values'][agent_idx]
            value_timesteps = timesteps[:len(values)]
            ax1.plot(value_timesteps, values, label=f'Agent {agent_idx}', alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Q-Value (V_i)')
    ax1.set_title('Agent Values Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    episode_timesteps = data.get('episode_timesteps', None)
    for agent_idx in range(num_agents):
        if agent_idx in data['episode_rewards'] and len(data['episode_rewards'][agent_idx]) > 0:
            rewards = data['episode_rewards'][agent_idx]
            max_points = 1000
            if len(rewards) > max_points:
                step = len(rewards) // max_points
                indices = np.arange(0, len(rewards), step)
                rewards_downsampled = rewards[indices]
                
                if episode_timesteps is not None and len(episode_timesteps) == len(rewards):
                    timesteps_downsampled = episode_timesteps[indices]
                    ax2.plot(timesteps_downsampled, rewards_downsampled, 
                            label=f'Agent {agent_idx}', alpha=0.6, linewidth=0.5, markersize=1)
                else:
                    ax2.plot(indices, rewards_downsampled, 
                            label=f'Agent {agent_idx}', alpha=0.6, linewidth=0.5, markersize=1)
            else:
                if episode_timesteps is not None and len(episode_timesteps) == len(rewards):
                    ax2.plot(episode_timesteps, rewards, label=f'Agent {agent_idx}', alpha=0.6, linewidth=0.5, markersize=1)
                else:
                    ax2.plot(range(len(rewards)), rewards, label=f'Agent {agent_idx}', alpha=0.6, linewidth=0.5, markersize=1)
            if len(rewards) > 50:
                window = min(100, len(rewards) // 10)
                if episode_timesteps is not None and len(episode_timesteps) == len(rewards):
                    rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    timesteps_ma = episode_timesteps[window-1:]
                    ax2.plot(timesteps_ma, rewards_ma, 
                            label=f'Agent {agent_idx} (MA)', alpha=0.9, linewidth=2)
                else:
                    rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    timesteps_ma = np.arange(window-1, len(rewards))
                    ax2.plot(timesteps_ma, rewards_ma, 
                            label=f'Agent {agent_idx} (MA)', alpha=0.9, linewidth=2)
    
    ax2.set_xlabel('Training Step' if episode_timesteps is not None else 'Episode Number')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Episode Rewards Over Time (with Moving Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for agent_idx in range(num_agents):
        if agent_idx in data['policy_params']:
            params = data['policy_params'][agent_idx]
            param_norms = np.linalg.norm(params, axis=1)
            ax3.plot(timesteps[:len(param_norms)], param_norms, label=f'Agent {agent_idx}', alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Policy Parameter Norm ||θ_i||')
    ax3.set_title('Policy Parameter Magnitude Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for agent_idx in range(num_agents):
        if agent_idx in data['policy_params']:
            params = data['policy_params'][agent_idx]
            if len(params) > 1:
                param_changes = np.linalg.norm(np.diff(params, axis=0), axis=1)
                change_timesteps = timesteps[1:len(param_changes)+1]
                ax4.plot(change_timesteps, param_changes, label=f'Agent {agent_idx}', alpha=0.7)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Policy Parameter Change ||Δθ_i||')
    ax4.set_title('Policy Parameter Change Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plot_path = os.path.join(save_dir, 'trajectory_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
    else:
        plt.show()


def analyze_spillover_correlations(data):
    print("\n" + "="*60)
    print("SPILLOVER CORRELATION ANALYSIS")
    print("="*60)
    
    num_agents = data['num_agents']
    timesteps = data['timesteps']
    
    print("\n1. Policy Parameter Change Correlations:")
    print("   (How correlated are policy changes between agents?)")
    
    param_changes = {}
    for agent_idx in range(num_agents):
        if agent_idx in data['policy_params']:
            params = data['policy_params'][agent_idx]
            if len(params) > 1:
                param_changes[agent_idx] = np.diff(params, axis=0)
    
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            if i in param_changes and j in param_changes:
                changes_i = param_changes[i].flatten()
                changes_j = param_changes[j].flatten()
                min_len = min(len(changes_i), len(changes_j))
                if min_len > 1:
                    corr = np.corrcoef(changes_i[:min_len], changes_j[:min_len])[0, 1]
                    print(f"   Agent {i} ↔ Agent {j}: {corr:.4f}")
    
    print("\n2. Value Change Correlations:")
    print("   (How correlated are value changes between agents?)")
    
    value_changes = {}
    for agent_idx in range(num_agents):
        if agent_idx in data['agent_values']:
            values = data['agent_values'][agent_idx]
            if len(values) > 1:
                value_changes[agent_idx] = np.diff(values)
    
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            if i in value_changes and j in value_changes:
                changes_i = value_changes[i]
                changes_j = value_changes[j]
                min_len = min(len(changes_i), len(changes_j))
                if min_len > 1:
                    corr = np.corrcoef(changes_i[:min_len], changes_j[:min_len])[0, 1]
                    print(f"   Agent {i} ↔ Agent {j}: {corr:.4f}")

    print("\n3. Lagged Correlations (Spillover Detection):")
    print("   (Does Agent i's change predict Agent j's future change?)")
    
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j and i in param_changes and j in value_changes:
                changes_i = param_changes[i]
                changes_j = value_changes[j]
                min_len = min(len(changes_i), len(changes_j))
                
                if min_len > 2:
                    param_change_magnitude = np.linalg.norm(changes_i, axis=1)[:min_len-1]
                    value_changes_j = changes_j[1:min_len]
                    
                    if len(param_change_magnitude) > 1 and len(value_changes_j) > 1:
                        corr = np.corrcoef(param_change_magnitude, value_changes_j)[0, 1]
                        if abs(corr) > 0.1:  # Only show significant correlations
                            print(f"   Agent {i} param change → Agent {j} value change: {corr:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory data for spillover detection")
    parser.add_argument("--trajectory-file", type=str, required=True,
                       help="Path to trajectory_data.pkl file")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--save-plots", type=str, default=None,
                       help="Directory to save plots (if not specified, plots are displayed)")
    
    args = parser.parse_args()

    print(f"Loading trajectory data from: {args.trajectory_file}")
    data = load_trajectory_data(args.trajectory_file)

    print_data_summary(data)

    analyze_spillover_correlations(data)

    if args.plot:
        save_dir = args.save_plots if args.save_plots else os.path.dirname(args.trajectory_file)
        plot_trajectories(data, save_dir=save_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

