import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime

from srmapg import SRMAPG
from replay_buffer import ReplayBuffer
from env_utils import get_env_info, create_env
from logger import Logger
from evaluator import evaluate
from trajectory_collector import TrajectoryCollector
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Train SR-MAPG (Spillover-Resistant MADDPG)")
    parser.add_argument("--env-name", type=str, default="simple_spread_v3")
    parser.add_argument("--algo", type=str, default="SRMAPG")
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
    
    # SR-MAPG specific arguments
    parser.add_argument("--lambda-inf", type=float, default=0.1,
                       help="Weight for influence/spillover penalty")
    parser.add_argument("--lambda-kl", type=float, default=0.01,
                       help="Weight for KL regularization")
    parser.add_argument("--spsa-epsilon", type=float, default=0.01,
                       help="Perturbation size for SPSA Jacobian estimation")
    parser.add_argument("--influence-update-interval", type=int, default=1000,
                       help="How often to update influence matrix (in steps)")
    parser.add_argument("--load-influence-matrix", type=str, default=None,
                       help="Path to pre-computed influence matrix (.npy or .pkl file)")
    parser.add_argument("--freeze-influence-matrix", action="store_true",
                       help="Freeze influence matrix (don't update during training)")
    parser.add_argument("--influence-update-alpha", type=float, default=0.1,
                       help="Exponential moving average coefficient for influence updates (0.1 = 10% new, 90% old)")
    parser.add_argument("--jacobian-update-interval", type=int, default=100,
                       help="Update Jacobian every N steps (default: 100, set higher for faster training)")
    parser.add_argument("--disable-influence-reg", action="store_true",
                       help="Disable influence regularization (use standard MADDPG)")
    
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
    
    # Initialize SR-MAPG
    srmapg = SRMAPG(
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
        lambda_inf=args.lambda_inf,
        lambda_kl=args.lambda_kl,
        spsa_epsilon=args.spsa_epsilon,
        influence_update_interval=args.influence_update_interval,
        use_influence_regularization=not args.disable_influence_reg,
        freeze_influence_matrix=args.freeze_influence_matrix,
        influence_update_alpha=args.influence_update_alpha
    )
    
    srmapg.jacobian_update_interval = args.jacobian_update_interval
    
    if args.load_influence_matrix:
        try:
            if args.load_influence_matrix.endswith('.npy'):
                S_operator = np.load(args.load_influence_matrix)
            elif args.load_influence_matrix.endswith('.pkl'):
                with open(args.load_influence_matrix, 'rb') as f:
                    results = pickle.load(f)

                if 'temporal_k_hops' in results:
                    temporal_k_hops = results['temporal_k_hops']
                    if 'temporal_spillover_operator' in temporal_k_hops:
                        S_operator = temporal_k_hops['temporal_spillover_operator']
                        print("Loaded temporal spillover operator from .pkl file")
                    elif 'temporal_windows' in temporal_k_hops:
                        windows = temporal_k_hops['temporal_windows']
                        if len(windows) > 0:
                            first_window = windows[sorted(windows.keys())[0]]
                            A = first_window['influence_matrix']
                            S_operator = A.astype(float)
                            if np.sum(S_operator) > 0:
                                S_operator = S_operator / np.sum(S_operator)
                            print("Loaded influence matrix from first temporal window")
                        else:
                            raise ValueError("No temporal windows found")
                    else:
                        raise ValueError("No temporal spillover operator found")
                elif 'influence_matrix' in results:
                    A = results['influence_matrix']
                    S_operator = A.astype(float)
                    if np.sum(S_operator) > 0:
                        S_operator = S_operator / np.sum(S_operator)
                    print("Loaded binary influence matrix from .pkl file")
                else:
                    raise ValueError("No influence matrix found in .pkl file")
            else:
                S_operator = np.load(args.load_influence_matrix)
            
            srmapg.set_influence_matrix(S_operator, freeze=args.freeze_influence_matrix)
            print(f"Loaded influence matrix from: {args.load_influence_matrix}")
            print(f"Influence matrix shape: {S_operator.shape}")
            print(f"Influence matrix:\n{S_operator}")
            if args.freeze_influence_matrix:
                print("Influence matrix is FROZEN - will not update during training")
            else:
                print(f"Influence matrix will update every {args.influence_update_interval} steps")
                print(f" Update alpha: {args.influence_update_alpha} (exponential moving average)")
            print(f"Matrix sum: {np.sum(S_operator):.4f}")
        except Exception as e:
            print(f"Warning: Could not load influence matrix: {e}")
            print("Will estimate influence matrix during training")
    
    buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_agents=num_agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes
    )
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
    
    noise_scale = args.noise_scale
    best_score = -float('inf')
    episode_rewards = np.zeros(num_agents)
    
    print(f"\n{'='*60}")
    print(f"Training {args.algo} on {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Spillover regularization: {'Enabled' if not args.disable_influence_reg else 'Disabled'}")
    if not args.disable_influence_reg:
        print(f"  Lambda_inf (spillover penalty): {args.lambda_inf}")
        print(f"  Lambda_kl (KL regularization): {args.lambda_kl}")
        print(f"  Influence update interval: {args.influence_update_interval}")
        print(f"  Freeze influence matrix: {args.freeze_influence_matrix}")
        if not args.freeze_influence_matrix:
            print(f"  Influence update alpha: {args.influence_update_alpha} (EMA coefficient)")
    print(f"{'='*60}\n")
    print("Running initial evaluation")
    avg_rewards, success_rate = evaluate(env_eval, srmapg, agents, logger, 0, env_name=args.env_name)

    observations, _ = env.reset()
    
    for global_step in tqdm(range(1, args.total_timesteps + 1), desc="Training"):
        states = [np.array(observations[agent], dtype=np.float32) for agent in agents]
        actions = srmapg.act(states, noise_scale=noise_scale)
        actions_dict = {agent: action for agent, action in zip(agents, actions)}
        next_observations, rewards, terminations, truncations, _ = env.step(actions_dict)
        dones = [terminations[agent] or truncations[agent] for agent in agents]
        done = any(dones)

        rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
        next_states = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
        dones_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)

        buffer.add(states, actions, rewards_array, next_states, dones_array)
        observations = next_observations
        episode_rewards += rewards_array

        if trajectory_collector is not None:
            trajectory_collector.collect(
                timestep=global_step,
                maddpg=srmapg,
                states=states,
                episode_rewards=episode_rewards if done or (global_step % args.max_steps == 0) else None
            )
        
        if global_step > args.warmup_steps and global_step % args.update_every == 0:
            for agent_idx in range(num_agents):
                batch = buffer.sample()
                critic_loss, actor_loss, spillover_loss, kl_loss = srmapg.learn(
                    batch, agent_idx, global_step=global_step
                )
                
                logger.log_scalar(f'{agents[agent_idx]}/critic_loss', critic_loss, global_step)
                logger.log_scalar(f'{agents[agent_idx]}/actor_loss', actor_loss, global_step)
                if not args.disable_influence_reg:
                    logger.log_scalar(f'{agents[agent_idx]}/spillover_loss', spillover_loss, global_step)
                    logger.log_scalar(f'{agents[agent_idx]}/kl_loss', kl_loss, global_step)
            
            srmapg.update_targets()

            if not args.disable_influence_reg and global_step % args.influence_update_interval == 0:
                S = srmapg.get_influence_matrix()
                logger.log_scalar('influence/matrix_sum', np.sum(S), global_step)
                logger.log_scalar('influence/matrix_max', np.max(S), global_step)
        
        if done or (global_step % args.max_steps == 0):
            for i, agent in enumerate(agents):
                logger.log_scalar(f'{agent}/episode_reward', episode_rewards[i], global_step)
            logger.log_scalar('train/total_reward', np.sum(episode_rewards), global_step)
            logger.log_scalar('train/noise_scale', noise_scale, global_step)
            
            observations, _ = env.reset()
            episode_rewards = np.zeros(num_agents)
        
        if global_step % args.eval_interval == 0 or global_step == args.total_timesteps:
            srmapg.save(model_path)
            
            avg_rewards, success_rate = evaluate(env_eval, srmapg, agents, logger, global_step, env_name=args.env_name)
            score = np.sum(avg_rewards)
            
            if score > best_score:
                best_score = score
                srmapg.save(best_model_path)
                print(f"New best score: {score:.2f} (success rate: {success_rate:.1f}%)")

            if not args.disable_influence_reg:
                S = srmapg.get_influence_matrix()
                influence_path = os.path.join(save_dir, f"influence_matrix_step_{global_step}.npy")
                np.save(influence_path, S)
    
    srmapg.save(model_path)

    if trajectory_collector is not None:
        trajectory_collector.save("trajectory_data.pkl")
        summary = trajectory_collector.get_trajectory_summary()
        print(f"\nTrajectory Collection Summary:")
        print(f"  - Total snapshots: {summary['num_snapshots']}")
        print(f"  - Timestep range: {summary['timestep_range']}")

    if not args.disable_influence_reg:
        S_final = srmapg.get_influence_matrix()
        np.save(os.path.join(save_dir, "influence_matrix_final.npy"), S_final)
        print(f"\nFinal influence matrix saved to: {os.path.join(save_dir, 'influence_matrix_final.npy')}")
        print(f"Final influence matrix:\n{S_final}")
    
    env.close()
    env_eval.close()
    logger.close()
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    return experiment_name


if __name__ == "__main__":
    args = parse_args()
    train(args)

