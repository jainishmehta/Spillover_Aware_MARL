import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob


def load_trajectory_data(filepath):
    """Load trajectory data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_trajectory_data(data, save_dir=None, agent_names=None):
    """
    Plot trajectory data including agent values, episode rewards, and policy parameters.
    
    Args:
        data: Dictionary containing trajectory data
        save_dir: Directory to save plots (if None, displays plots)
        agent_names: Optional list of agent names (default: Agent 0, Agent 1, ...)
    """
    num_agents = data['num_agents']
    
    if agent_names is None:
        agent_names = [f'Agent {i}' for i in range(num_agents)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Agent Values over time
    ax1 = plt.subplot(2, 2, 1)
    if 'agent_values' in data and len(data['agent_values']) > 0:
        timesteps = data.get('timesteps', None)
        for agent_idx in range(num_agents):
            if agent_idx in data['agent_values']:
                values = data['agent_values'][agent_idx]
                if timesteps is not None and len(timesteps) == len(values):
                    ax1.plot(timesteps, values, label=agent_names[agent_idx], linewidth=1.5, alpha=0.8)
                else:
                    ax1.plot(values, label=agent_names[agent_idx], linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Q-Value Estimate', fontsize=12)
        ax1.set_title('Agent Value Estimates Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Agent values not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Agent Value Estimates', fontsize=14, fontweight='bold')
    
    # Plot 2: Episode Rewards over time
    ax2 = plt.subplot(2, 2, 2)
    if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
        episode_timesteps = data.get('episode_timesteps', None)
        for agent_idx in range(num_agents):
            if agent_idx in data['episode_rewards']:
                rewards = data['episode_rewards'][agent_idx]
                if episode_timesteps is not None and len(episode_timesteps) == len(rewards):
                    ax2.plot(episode_timesteps, rewards, label=agent_names[agent_idx], 
                            linewidth=1.5, alpha=0.8, marker='o', markersize=2)
                else:
                    ax2.plot(rewards, label=agent_names[agent_idx], 
                            linewidth=1.5, alpha=0.8, marker='o', markersize=2)
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Episode Reward', fontsize=12)
        ax2.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Episode rewards not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    
    # Plot 3: Policy Parameter Norms over time
    ax3 = plt.subplot(2, 2, 3)
    if 'policy_params' in data and len(data['policy_params']) > 0:
        timesteps = data.get('timesteps', None)
        for agent_idx in range(num_agents):
            if agent_idx in data['policy_params']:
                params = data['policy_params'][agent_idx]
                # Compute L2 norm of parameters for each timestep
                param_norms = np.linalg.norm(params, axis=1)
                if timesteps is not None and len(timesteps) == len(param_norms):
                    ax3.plot(timesteps, param_norms, label=agent_names[agent_idx], 
                            linewidth=1.5, alpha=0.8)
                else:
                    ax3.plot(param_norms, label=agent_names[agent_idx], 
                            linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('Policy Parameter L2 Norm', fontsize=12)
        ax3.set_title('Policy Parameter Norms Over Time', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Policy parameters not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Policy Parameters', fontsize=14, fontweight='bold')
    
    # Plot 4: Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = "Trajectory Data Summary\n" + "="*40 + "\n\n"
    summary_text += f"Number of agents: {num_agents}\n"
    summary_text += f"Collection interval: {data.get('collect_interval', 'N/A')} steps\n\n"
    
    if 'timesteps' in data and len(data['timesteps']) > 0:
        timesteps = data['timesteps']
        summary_text += f"Timestep range: {timesteps[0]:,} - {timesteps[-1]:,}\n"
        summary_text += f"Total snapshots: {len(timesteps)}\n\n"
    
    if 'agent_values' in data:
        summary_text += "Agent Values:\n"
        for agent_idx in range(num_agents):
            if agent_idx in data['agent_values']:
                values = data['agent_values'][agent_idx]
                summary_text += f"  {agent_names[agent_idx]}: {len(values)} points, "
                summary_text += f"mean={np.mean(values):.3f}, std={np.std(values):.3f}\n"
        summary_text += "\n"
    
    if 'episode_rewards' in data:
        summary_text += "Episode Rewards:\n"
        for agent_idx in range(num_agents):
            if agent_idx in data['episode_rewards']:
                rewards = data['episode_rewards'][agent_idx]
                summary_text += f"  {agent_names[agent_idx]}: {len(rewards)} episodes, "
                summary_text += f"mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}\n"
        summary_text += "\n"
    
    if 'policy_params' in data:
        summary_text += "Policy Parameters:\n"
        for agent_idx in range(num_agents):
            if agent_idx in data['policy_params']:
                params = data['policy_params'][agent_idx]
                summary_text += f"  {agent_names[agent_idx]}: {params.shape[0]} snapshots, "
                summary_text += f"{params.shape[1]} parameters\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'trajectory_plots.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory data from MADDPG training")
    parser.add_argument("--trajectory-file", type=str, default=None,
                       help="Path to trajectory_data.pkl file")
    parser.add_argument("--experiment-dir", type=str, default=None,
                       help="Path to experiment directory (will look for trajectory_data.pkl)")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save plots (if None, displays plots)")
    parser.add_argument("--agent-names", type=str, default=None,
                       help="Comma-separated list of agent names (e.g., 'adversary_0,agent_0,agent_1')")
    
    args = parser.parse_args()
    
    # Determine trajectory file path
    trajectory_file = args.trajectory_file
    
    if trajectory_file is None:
        if args.experiment_dir:
            trajectory_file = os.path.join(args.experiment_dir, 'trajectory_data.pkl')
        else:
            # Look for the most recent trajectory file
            pattern = "runs/*/trajectory_data.pkl"
            files = glob.glob(pattern)
            if files:
                # Sort by modification time, get most recent
                files.sort(key=os.path.getmtime, reverse=True)
                trajectory_file = files[0]
                print(f"Using most recent trajectory file: {trajectory_file}")
            else:
                print("Error: No trajectory file found. Please specify --trajectory-file or --experiment-dir")
                return
    
    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file not found: {trajectory_file}")
        return
    
    print(f"Loading trajectory data from: {trajectory_file}")
    data = load_trajectory_data(trajectory_file)
    
    # Parse agent names if provided
    agent_names = None
    if args.agent_names:
        agent_names = [name.strip() for name in args.agent_names.split(',')]
    
    # Determine save directory
    save_dir = args.save_dir
    if save_dir is None and args.experiment_dir:
        save_dir = args.experiment_dir
    
    print("Generating plots...")
    plot_trajectory_data(data, save_dir=save_dir, agent_names=agent_names)
    print("Done!")


if __name__ == "__main__":
    main()


