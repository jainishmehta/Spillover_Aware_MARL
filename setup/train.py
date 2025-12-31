import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime

from maddpg import MADDPG
from replay_buffer import ReplayBuffer
from env_utils import get_env_info, create_env
from logger import Logger
from evaluator import evaluate
from trajectory_collector import TrajectoryCollector


def apply_reward_shaping(rewards_dict, agent_names, env_name):
    shaped_rewards = rewards_dict.copy()
    
    if env_name == 'simple_adversary_v3':
        for agent_name in agent_names:
            if 'adversary' in agent_name.lower():
                if shaped_rewards[agent_name] < -10: 
                    shaped_rewards[agent_name] *= 1.0 
                elif shaped_rewards[agent_name] > -5: 
                    shaped_rewards[agent_name] += 0.5
                if shaped_rewards[agent_name] > 0:
                    shaped_rewards[agent_name] += 0.2
    
    return shaped_rewards


def get_curriculum_difficulty(global_step, total_timesteps, initial_noise=0.5, final_noise=0.1):
    progress = min(1.0, global_step / (total_timesteps * 0.5))  # Curriculum over first 50% of training
    noise_multiplier = initial_noise + (final_noise - initial_noise) * progress
    return noise_multiplier

def parse_args():
    parser = argparse.ArgumentParser(description="Train MADDPG")
    parser.add_argument("--env-name", type=str, default="simple_spread_v3")
    parser.add_argument("--algo", type=str, default="MADDPG")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6))
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--warmup-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=2e-3)
    parser.add_argument("--hidden-sizes", type=str, default="64,64")
    parser.add_argument("--update-every", type=int, default=15)
    parser.add_argument("--noise-scale", type=float, default=0.3)
    parser.add_argument("--min-noise", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--collect-trajectory", action="store_true", 
                       help="Enable trajectory data collection for spillover analysis")
    parser.add_argument("--trajectory-interval", type=int, default=1000,
                       help="Interval (in steps) for collecting trajectory data")
    parser.add_argument("--reward-scale", type=float, default=0.1,
                       help="Scale factor for rewards (helps stabilize Q-values)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="L2 regularization (weight decay) to prevent large parameters")
    parser.add_argument("--grad-clip-norm", type=float, default=0.5,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--adversary-actor-lr", type=float, default=None,
                       help="Learning rate for adversary actor (lower than good agents, e.g., 5e-4)")
    parser.add_argument("--adversary-critic-lr", type=float, default=None,
                       help="Learning rate for adversary critic (lower than good agents)")
    parser.add_argument("--separate-buffers", action="store_true",
                       help="Use separate replay buffers for adversary and good agents")
    parser.add_argument("--curriculum-learning", action="store_true",
                       help="Enable curriculum learning (progressive difficulty)")
    parser.add_argument("--reward-shaping", action="store_true",
                       help="Enable reward shaping for better learning signal")
    
    return parser.parse_args()

def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.algo}_{args.env_name}_{timestamp}"
    logger = Logger(experiment_name, args.env_name)
    logger.log_hyperparameters(vars(args))
    
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        args.env_name, args.max_steps
    )
    
    env = create_env(args.env_name, args.max_steps)
    env_eval = create_env(args.env_name, args.max_steps)

    hidden_sizes = tuple(map(int, args.hidden_sizes.split(',')))
    
    # Identify adversary agents (for simple_adversary_v3)
    agent_types = []
    adversary_indices = []
    for i, agent_name in enumerate(agents):
        if 'adversary' in agent_name.lower():
            agent_types.append('adversary')
            adversary_indices.append(i)
        else:
            agent_types.append('agent')

    actor_lrs = [args.actor_lr] * num_agents
    critic_lrs = [args.critic_lr] * num_agents
    
    if args.adversary_actor_lr is not None or args.adversary_critic_lr is not None:
        for idx in adversary_indices:
            if args.adversary_actor_lr is not None:
                actor_lrs[idx] = args.adversary_actor_lr
            if args.adversary_critic_lr is not None:
                critic_lrs[idx] = args.adversary_critic_lr
        print(f"\nPer-agent learning rates:")
        for i, agent_name in enumerate(agents):
            print(f"  {agent_name}: actor_lr={actor_lrs[i]:.6f}, critic_lr={critic_lrs[i]:.6f}")
    
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=hidden_sizes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        action_low=action_low,
        action_high=action_high,
        reward_scale=args.reward_scale,
        weight_decay=args.weight_decay,
        actor_lrs=actor_lrs,
        critic_lrs=critic_lrs,
        grad_clip_norm=args.grad_clip_norm,
        agent_types=agent_types
    )
    # Create replay buffers (separate for adversary if requested)
    if args.separate_buffers and len(adversary_indices) > 0:
        print(f"\nUsing separate replay buffers:")
        buffer_adversary = ReplayBuffer(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            num_agents=len(adversary_indices),
            state_sizes=[state_sizes[i] for i in adversary_indices],
            action_sizes=[action_sizes[i] for i in adversary_indices]
        )
        good_agent_indices = [i for i in range(num_agents) if i not in adversary_indices]
        buffer_good = ReplayBuffer(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            num_agents=len(good_agent_indices),
            state_sizes=[state_sizes[i] for i in good_agent_indices],
            action_sizes=[action_sizes[i] for i in good_agent_indices]
        )
        print(f"  Adversary buffer: {len(adversary_indices)} agents")
        print(f"  Good agents buffer: {len(good_agent_indices)} agents")
        buffer = None  # Will use separate buffers
    else:
        buffer = ReplayBuffer(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            num_agents=num_agents,
            state_sizes=state_sizes,
            action_sizes=action_sizes
        )
        buffer_adversary = None
        buffer_good = None
    save_dir = os.path.join("runs", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pt")
    best_model_path = os.path.join(save_dir, "best_model.pt")

    trajectory_collector = None
    if args.collect_trajectory:
        trajectory_collector = TrajectoryCollector(
            num_agents=num_agents,
            save_dir=save_dir,
            collect_interval=args.trajectory_interval
        )
        print(f"Trajectory collection enabled (interval: {args.trajectory_interval} steps)")

    # Curriculum learning: adjust noise scale over time
    if args.curriculum_learning:
        initial_noise = args.noise_scale
        final_noise = args.min_noise
        print(f"\nCurriculum learning enabled: noise will decrease from {initial_noise} to {final_noise}")
    else:
        initial_noise = args.noise_scale
        final_noise = args.noise_scale
    
    noise_scale = args.noise_scale
    best_score = -float('inf')
    episode_rewards = np.zeros(num_agents)

    print(f"\n{'='*60}")
    print(f"Training {args.algo} on {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")
    avg_rewards, success_rate = evaluate(env_eval, maddpg, agents, logger, 0)

    observations, _ = env.reset()
    
    for global_step in tqdm(range(1, args.total_timesteps + 1), desc="Training"):
        # Curriculum learning: adjust noise scale
        if args.curriculum_learning:
            noise_scale = get_curriculum_difficulty(global_step, args.total_timesteps, 
                                                    initial_noise, final_noise)
            noise_scale = max(noise_scale, args.min_noise)  # Don't go below min_noise
        
        states = [np.array(observations[agent], dtype=np.float32) for agent in agents]
        actions = maddpg.act(states, noise_scale=noise_scale)
        actions_dict = {agent: action for agent, action in zip(agents, actions)}
        next_observations, rewards, terminations, truncations, _ = env.step(actions_dict)
        dones = [terminations[agent] or truncations[agent] for agent in agents]
        done = any(dones)

        # Apply reward shaping if enabled
        if args.reward_shaping:
            rewards = apply_reward_shaping(rewards, agents, args.env_name)
        
        rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
        next_states = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
        dones_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)

        # Add to buffers (separate or shared)
        if args.separate_buffers and buffer_adversary is not None:
            # Split states, actions, rewards for separate buffers
            adv_states = [states[i] for i in adversary_indices]
            adv_actions = [actions[i] for i in adversary_indices]
            adv_rewards = rewards_array[adversary_indices]
            adv_next_states = [next_states[i] for i in adversary_indices]
            adv_dones = dones_array[adversary_indices]
            
            good_indices = [i for i in range(num_agents) if i not in adversary_indices]
            good_states = [states[i] for i in good_indices]
            good_actions = [actions[i] for i in good_indices]
            good_rewards = rewards_array[good_indices]
            good_next_states = [next_states[i] for i in good_indices]
            good_dones = dones_array[good_indices]
            
            buffer_adversary.add(adv_states, adv_actions, adv_rewards, adv_next_states, adv_dones)
            buffer_good.add(good_states, good_actions, good_rewards, good_next_states, good_dones)
        else:
            buffer.add(states, actions, rewards_array, next_states, dones_array)
        observations = next_observations
        episode_rewards += rewards_array

        if trajectory_collector is not None:
            trajectory_collector.collect(
                timestep=global_step,
                maddpg=maddpg,
                states=states,
                episode_rewards=episode_rewards if done or (global_step % args.max_steps == 0) else None
            )
        
        if global_step > args.warmup_steps and global_step % args.update_every == 0:
            for agent_idx in range(num_agents):
                batch = buffer.sample()
                critic_loss, actor_loss = maddpg.learn(batch, agent_idx)
                
                logger.log_scalar(f'{agents[agent_idx]}/critic_loss', critic_loss, global_step)
                logger.log_scalar(f'{agents[agent_idx]}/actor_loss', actor_loss, global_step)
            
            maddpg.update_targets()
        if done or (global_step % args.max_steps == 0):
            for i, agent in enumerate(agents):
                logger.log_scalar(f'{agent}/episode_reward', episode_rewards[i], global_step)
            logger.log_scalar('train/total_reward', np.sum(episode_rewards), global_step)
            logger.log_scalar('train/noise_scale', noise_scale, global_step)
            observations, _ = env.reset()
            episode_rewards = np.zeros(num_agents)
        if global_step % args.eval_interval == 0 or global_step == args.total_timesteps:
            maddpg.save(model_path)
            
            avg_rewards, success_rate = evaluate(env_eval, maddpg, agents, logger, global_step)
            score = np.sum(avg_rewards)
            
            if score > best_score:
                best_score = score
                maddpg.save(best_model_path)
                print(f"New best score: {score:.2f} (success rate: {success_rate:.1f}%)")
    maddpg.save(model_path)

    if trajectory_collector is not None:
        trajectory_collector.save("trajectory_data.pkl")
        summary = trajectory_collector.get_trajectory_summary()
        print(f"  - Total snapshots: {summary['num_snapshots']}")
        print(f"  - Timestep range: {summary['timestep_range']}")

    env.close()
    env_eval.close()
    logger.close()
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    return experiment_name


if __name__ == "__main__":
    args = parse_args()
    train(args)