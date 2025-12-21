import numpy as np
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def load_trajectory_data(filepath):
    """Load trajectory data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_trajectories(data, metric='agent_values', max_lag=5):
    """    
    Args:
        data: Trajectory data dictionary
        metric: Which metric to use ('agent_values', 'policy_params', 'episode_rewards')
        max_lag: Maximum lag to consider
    
    Returns:
        trajectories: Array of shape (T, num_agents) with time series data
        timesteps: Array of timesteps
    """
    num_agents = data['num_agents']
    
    if metric == 'agent_values':
        trajectories = []
        for agent_idx in range(num_agents):
            if agent_idx in data['agent_values']:
                trajectories.append(data['agent_values'][agent_idx])
            else:
                raise ValueError(f"Missing agent_values for agent {agent_idx}")
        trajectories = np.array(trajectories).T
        
    elif metric == 'policy_params':
        trajectories = []
        for agent_idx in range(num_agents):
            if agent_idx in data['policy_params']:
                params = data['policy_params'][agent_idx]
                param_norms = np.linalg.norm(params, axis=1)
                trajectories.append(param_norms)
            else:
                raise ValueError(f"Missing policy_params for agent {agent_idx}")
        trajectories = np.array(trajectories).T  # Shape: (T, num_agents)
        
    elif metric == 'episode_rewards':
        trajectories = []
        episode_timesteps = data.get('episode_timesteps', None)
        for agent_idx in range(num_agents):
            if agent_idx in data['episode_rewards']:
                trajectories.append(data['episode_rewards'][agent_idx])
            else:
                raise ValueError(f"Missing episode_rewards for agent {agent_idx}")
        trajectories = np.array(trajectories).T
        
    else:
        raise ValueError(f"Unknown metric: {metric}")

    trajectories = np.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)

    if trajectories.shape[0] < max_lag + 10:
        raise ValueError(f"Not enough data points ({trajectories.shape[0]}) for max_lag={max_lag}")
    
    return trajectories


def estimate_influence_granger(trajectories, max_lag=5, alpha=0.05, method='fdr_bh'):
    """
    Args:
        trajectories: Array of shape (T, num_agents) with time series data
        max_lag: Maximum lag to consider in VAR model
        alpha: Significance level
        method: Multiple testing correction method ('fdr_bh' for Benjamini-Hochberg)
    
    Returns:
        A: Influence matrix where A[j,i] = 1 if agent i Granger-causes agent j
        p_values: Matrix of p-values for each pair
        test_statistics: Matrix of F-statistics for each pair
    """
    T, num_agents = trajectories.shape
    
    A = np.zeros((num_agents, num_agents), dtype=int)
    p_values = np.ones((num_agents, num_agents))
    test_statistics = np.zeros((num_agents, num_agents))
    
    print(f"\n{'='*60}")
    print("Granger Causality Analysis")
    print(f"{'='*60}")
    print(f"Data shape: {trajectories.shape}")
    print(f"Max lag: {max_lag}")
    print(f"Significance level: {alpha}")
    print(f"FDR correction method: {method}\n")
    
    try:
        model = VAR(trajectories)
        lag_order = model.select_order(maxlags=max_lag)
        optimal_lag = lag_order.selected_orders.get('aic', max_lag)
        optimal_lag = min(optimal_lag, max_lag)
        print(f"Optimal lag selected: {optimal_lag}")
        
        fitted_model = model.fit(optimal_lag)
        
        print("\nModel Diagnostics:")
        print(f"  - AIC: {fitted_model.aic:.4f}")
        print(f"  - BIC: {fitted_model.bic:.4f}")
        
        ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
        print(f"  - Ljung-Box p-value: {ljung_box['lb_pvalue'].iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"Warning: VAR model fitting failed: {e}")
        print("Using default lag order")
        optimal_lag = max_lag
        fitted_model = model.fit(optimal_lag)
    
    # Perform Granger causality tests for each pair
    print(f"\nTesting {num_agents * (num_agents - 1)} pairs for Granger causality...")
    
    all_p_values = []
    test_results = []
    
    for j in range(num_agents): 
        for i in range(num_agents):
            if i == j:
                continue
            try:
                test_result = fitted_model.test_causality(j, i, kind='f')
                
                p_val = test_result.pvalue
                f_stat = test_result.test_statistic
                
                p_values[j, i] = p_val
                test_statistics[j, i] = f_stat
                all_p_values.append(p_val)
                test_results.append((j, i, p_val, f_stat))
                
            except Exception as e:
                print(f"Warning: Granger test failed for {i} -> {j}: {e}")
                p_values[j, i] = 1.0
                test_statistics[j, i] = 0.0

    print(f"\nApplying FDR correction ({method})...")
    all_p_values = np.array(all_p_values)
    rejected, pvals_corrected, _, _ = multipletests(
        all_p_values, alpha=alpha, method=method
    )

    corrected_idx = 0
    for j in range(num_agents):
        for i in range(num_agents):
            if i == j:
                continue
            
            if rejected[corrected_idx]:
                A[j, i] = 1
                print(f"  Agent {i} -> Agent {j}: p={p_values[j,i]:.4f}, corrected_p={pvals_corrected[corrected_idx]:.4f} ✓")
            
            corrected_idx += 1
    
    print(f"\n{'='*60}")
    print(f"Influence Matrix (A[j,i] = 1 if i Granger-causes j):")
    print(f"{'='*60}")
    print(A)
    print(f"\nTotal causal relationships found: {np.sum(A)}")
    
    return A, p_values, test_statistics


def compute_influence_statistics(A, A_ground_truth=None):
    """
    Args:
        A: Estimated influence matrix
        A_ground_truth: Ground truth influence matrix (optional)
    
    Returns:
        stats: Dictionary with statistics
    """
    num_agents = A.shape[0]
    stats = {
        'num_causal_links': int(np.sum(A)),
        'total_possible_links': num_agents * (num_agents - 1),
        'sparsity': 1.0 - (np.sum(A) / (num_agents * (num_agents - 1))),
        'in_degree': np.sum(A, axis=0),
        'out_degree': np.sum(A, axis=1),
    }
    
    if A_ground_truth is not None:
        A_flat = A.flatten()
        A_gt_flat = A_ground_truth.flatten()

        mask = ~np.eye(num_agents, dtype=bool).flatten()
        A_flat = A_flat[mask]
        A_gt_flat = A_gt_flat[mask]
        
        tp = np.sum((A_flat == 1) & (A_gt_flat == 1))
        fp = np.sum((A_flat == 1) & (A_gt_flat == 0))
        fn = np.sum((A_flat == 0) & (A_gt_flat == 1))
        tn = np.sum((A_flat == 0) & (A_gt_flat == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        stats['precision'] = precision
        stats['recall'] = recall
        stats['f1_score'] = f1_score
        stats['accuracy'] = accuracy
        stats['tp'] = int(tp)
        stats['fp'] = int(fp)
        stats['fn'] = int(fn)
        stats['tn'] = int(tn)
    
    return stats


def visualize_influence_matrix(A, p_values=None, save_path=None):
    """
    Args:
        A: Influence matrix
        p_values: Optional p-value matrix for visualization
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    num_agents = A.shape[0]
    
    fig, axes = plt.subplots(1, 2 if p_values is not None else 1, figsize=(12, 5))
    if p_values is None:
        axes = [axes]
    sns.heatmap(A, annot=True, fmt='d', cmap='Reds', 
                xticklabels=[f'Agent {i}' for i in range(num_agents)],
                yticklabels=[f'Agent {i}' for i in range(num_agents)],
                ax=axes[0], cbar_kws={'label': 'Influence'})
    axes[0].set_xlabel('Cause (Agent i)')
    axes[0].set_ylabel('Effect (Agent j)')
    axes[0].set_title('Influence Matrix A[j,i]\n(1 if i Granger-causes j)')
    if p_values is not None:
        p_values_masked = p_values.copy()
        p_values_masked[np.eye(num_agents, dtype=bool)] = np.nan
        
        sns.heatmap(p_values_masked, annot=True, fmt='.3f', cmap='viridis_r',
                   xticklabels=[f'Agent {i}' for i in range(num_agents)],
                   yticklabels=[f'Agent {i}' for i in range(num_agents)],
                   ax=axes[1], cbar_kws={'label': 'p-value'}, vmin=0, vmax=1)
        axes[1].set_xlabel('Cause (Agent i)')
        axes[1].set_ylabel('Effect (Agent j)')
        axes[1].set_title('Granger Causality p-values')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nInfluence matrix visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate influence matrix using Granger Causality")
    parser.add_argument("--trajectory-file", type=str, required=True,
                       help="Path to trajectory_data.pkl file")
    parser.add_argument("--metric", type=str, default="agent_values",
                       choices=["agent_values", "policy_params", "episode_rewards"],
                       help="Which metric to use for analysis")
    parser.add_argument("--max-lag", type=int, default=5,
                       help="Maximum lag for VAR model")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level for FDR correction")
    parser.add_argument("--ground-truth", type=str, default=None,
                       help="Path to ground truth influence matrix (optional)")
    parser.add_argument("--save-results", type=str, default=None,
                       help="Path to save results (optional)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()

    print(f"Loading trajectory data from: {args.trajectory_file}")
    data = load_trajectory_data(args.trajectory_file)

    print(f"\nPreparing trajectories using metric: {args.metric}")
    trajectories = prepare_trajectories(data, metric=args.metric, max_lag=args.max_lag)

    A, p_values, test_statistics = estimate_influence_granger(
        trajectories, max_lag=args.max_lag, alpha=args.alpha
    )

    A_ground_truth = None
    if args.ground_truth:
        A_ground_truth = np.load(args.ground_truth)
        print(f"\nLoaded ground truth from: {args.ground_truth}")

    stats = compute_influence_statistics(A, A_ground_truth)
    
    print(f"\n{'='*60}")
    print("Influence Statistics")
    print(f"{'='*60}")
    print(f"Number of causal links: {stats['num_causal_links']}")
    print(f"Sparsity: {stats['sparsity']:.2%}")
    print(f"\nIn-degree (agents influenced by each agent):")
    for i in range(len(stats['in_degree'])):
        print(f"  Agent {i}: {stats['in_degree'][i]}")
    print(f"\nOut-degree (agents each agent influences):")
    for i in range(len(stats['out_degree'])):
        print(f"  Agent {i}: {stats['out_degree'][i]}")
    
    if A_ground_truth is not None:
        print(f"\nValidation Metrics (vs Ground Truth):")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")
        print(f"  F1-Score: {stats['f1_score']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}, TN: {stats['tn']}")

    if args.visualize:
        save_path = args.save_results.replace('.pkl', '_influence_matrix.png') if args.save_results else None
        visualize_influence_matrix(A, p_values, save_path=save_path)

    if args.save_results:
        results = {
            'influence_matrix': A,
            'p_values': p_values,
            'test_statistics': test_statistics,
            'statistics': stats,
            'metric': args.metric,
            'max_lag': args.max_lag,
            'alpha': args.alpha
        }
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()

