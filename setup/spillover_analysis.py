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
        trajectories = np.array(trajectories).T 
        
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
    
    # Get timesteps for temporal analysis
    timesteps = data.get('timesteps', np.arange(trajectories.shape[0]))
    
    return trajectories, timesteps


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
                print(f"  Agent {i} -> Agent {j}: p={p_values[j,i]:.4f}, corrected_p={pvals_corrected[corrected_idx]:.4f}")
            
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

def compute_k_hops_propagation_with_time(A, trajectories, timesteps, max_hops=3, time_windows=None):
    """
    Args:
        A: Direct influence matrix (from Granger causality)
        trajectories: Time series data (T, num_agents)
        timesteps: Array of timesteps when data was collected
        max_hops: Maximum number of hops to analyze
        time_windows: List of time window boundaries for analysis (e.g., [0, 50000, 100000, 150000])
    
    Returns:
        results: Dictionary with temporal k-hops analysis
    """
    T, num_agents = trajectories.shape
    
    print(f"\n{'='*60}")
    print(f"Temporal K-Hops Propagation Analysis")
    print(f"{'='*60}")
    print(f"Data points: {T}")
    print(f"Timestep range: {timesteps[0] if len(timesteps) > 0 else 'N/A'} to {timesteps[-1] if len(timesteps) > 0 else 'N/A'}")
    
    results = {
        'global_k_hops': None,
        'temporal_windows': {},
        'rolling_analysis': {}
    }

    print(f"\n{'='*60}")
    print("Global K-Hops Analysis (Full Time Period)")
    print(f"{'='*60}")
    global_k_hops = compute_k_hops_propagation(A, max_hops=max_hops)
    results['global_k_hops'] = global_k_hops

    if time_windows is None:
        num_windows = 3
        if len(timesteps) > 0:
            time_range = timesteps[-1] - timesteps[0]
            window_size = time_range / num_windows
            time_windows = [timesteps[0] + i * window_size for i in range(num_windows + 1)]
        else:
            # Use data indices if no timesteps
            window_size = T / num_windows
            time_windows = [int(i * window_size) for i in range(num_windows + 1)]
    
    print(f"\n{'='*60}")
    print("Temporal Window Analysis")
    print(f"{'='*60}")
    print(f"Time windows: {time_windows}")
    
    for window_idx in range(len(time_windows) - 1):
        window_start = time_windows[window_idx]
        window_end = time_windows[window_idx + 1]

        if len(timesteps) > 0:
            mask = (timesteps >= window_start) & (timesteps < window_end)
        else:
            mask = np.arange(T) >= int(window_start)
            mask = mask & (np.arange(T) < int(window_end))
        window_data = trajectories[mask]
        if len(window_data) < 50:
            print(f"\nWindow {window_idx + 1} ({window_start:.0f}-{window_end:.0f}): Insufficient data ({len(window_data)} points)")
            continue
        
        print(f"\nWindow {window_idx + 1} ({window_start:.0f}-{window_end:.0f}):")
        print(f"  Data points: {len(window_data)}")

        try:
            A_window, _, _ = estimate_influence_granger(
                window_data, 
                max_lag=min(5, len(window_data) // 20),  # Adaptive lag
                alpha=0.05
            )
            
            # Compute k-hops for this window
            k_hops_window = compute_k_hops_propagation(A_window, max_hops=max_hops)
            
            results['temporal_windows'][window_idx] = {
                'time_range': (window_start, window_end),
                'influence_matrix': A_window,
                'k_hops': k_hops_window,
                'num_data_points': len(window_data)
            }
            
            print(f"  Direct links: {np.sum(A_window)}")
            for k in range(2, max_hops + 1):
                indirect = k_hops_window['k_hops_matrices'][k]
                print(f"  {k}-hop indirect links: {np.sum(indirect)}")
        
        except Exception as e:
            print(f"  Error analyzing window: {e}")
            continue

    print(f"\n{'='*60}")
    print("Rolling Window Analysis")
    print(f"{'='*60}")
    
    window_size = min(100, T // 5)
    step_size = window_size // 2  # 50% overlap
    
    rolling_results = []
    
    for start_idx in range(0, T - window_size, step_size):
        end_idx = start_idx + window_size
        window_data = trajectories[start_idx:end_idx]
        
        if len(timesteps) > 0:
            window_times = (timesteps[start_idx], timesteps[end_idx-1])
        else:
            window_times = (start_idx, end_idx-1)
        
        try:
            A_rolling, _, _ = estimate_influence_granger(
                window_data,
                max_lag=min(3, len(window_data) // 15),
                alpha=0.05
            )
            
            direct_links = np.sum(A_rolling)
            rolling_results.append({
                'time_range': window_times,
                'data_range': (start_idx, end_idx),
                'influence_matrix': A_rolling,
                'direct_links': int(direct_links)
            })
        
        except Exception as e:
            continue
    
    results['rolling_analysis'] = {
        'window_size': window_size,
        'step_size': step_size,
        'results': rolling_results
    }
    
    print(f"Analyzed {len(rolling_results)} rolling windows")
    if len(rolling_results) > 0:
        direct_links_over_time = [r['direct_links'] for r in rolling_results]
        print(f"  Mean direct links: {np.mean(direct_links_over_time):.2f}")
        print(f"  Std direct links: {np.std(direct_links_over_time):.2f}")
        print(f"  Range: [{np.min(direct_links_over_time)}, {np.max(direct_links_over_time)}]")
    
    return results


def compute_k_hops_propagation(A, max_hops=3):
    """
    Args:
        A: Direct influence matrix (binary, A[j,i] = 1 if i directly influences j)
        max_hops: Maximum number of hops to analyze
    
    Returns:
        k_hops_matrices: Dictionary with k-hop influence matrices
        k_hops_paths: Dictionary with paths for each k-hop relationship
    """
    num_agents = A.shape[0]
    
    print(f"\n{'='*60}")
    print(f"K-Hops Propagation Analysis (up to {max_hops} hops)")
    print(f"{'='*60}")
    
    k_hops_matrices = {}
    k_hops_paths = {}

    k_hops_matrices[1] = A.copy()
    print(f"\n1-hop (Direct) Influence:")
    print(f"  Total direct links: {np.sum(A)}")

    for k in range(2, max_hops + 1):
        k_hop_matrix = np.linalg.matrix_power(A.astype(float), k)
        k_hop_binary = (k_hop_matrix > 0).astype(int)

        np.fill_diagonal(k_hop_binary, 0)
        k_hops_matrices[k] = k_hop_binary
        
        num_k_hops = np.sum(k_hop_binary)
        print(f"\n{k}-hop (Indirect) Influence:")
        print(f"  Total {k}-hop paths: {num_k_hops}")
    
        paths = []
        for j in range(num_agents):
            for i in range(num_agents):
                if k_hop_binary[j, i] == 1 and i != j:
                    paths.append((i, j, k))

        k_hops_paths[k] = paths
        
        if num_k_hops > 0:
            print(f"  Example {k}-hop relationships:")
            for i, j, _ in paths[:5]:
                print(f"    Agent {i} → ... → Agent {j} ({k} hops)")
    
    print(f"\n{'='*60}")
    print("Cumulative Influence (all paths up to k hops)")
    print(f"{'='*60}")
    
    cumulative_matrices = {}
    for k in range(1, max_hops + 1):
        cumulative = np.zeros_like(A)
        for h in range(1, k + 1):
            cumulative = np.logical_or(cumulative, k_hops_matrices[h]).astype(int)
        
        cumulative_matrices[k] = cumulative
        num_total = np.sum(cumulative)
        print(f"\nUp to {k}-hops: {num_total} total influence relationships")
        
        # Show new relationships discovered at this hop level
        if k > 1:
            prev_cumulative = cumulative_matrices[k-1]
            new_relationships = cumulative - prev_cumulative
            num_new = np.sum(new_relationships)
            if num_new > 0:
                print(f"  New relationships at {k}-hop level: {num_new}")
    
    return {
        'k_hops_matrices': k_hops_matrices,
        'k_hops_paths': k_hops_paths,
        'cumulative_matrices': cumulative_matrices,
        'max_hops': max_hops
    }


def analyze_indirect_spillover(A, k_hops_results):
    """
    Args:
        A: Direct influence matrix
        k_hops_results: Results from compute_k_hops_propagation
    
    Returns:
        analysis: Dictionary with indirect spillover analysis
    """
    num_agents = A.shape[0]
    
    print(f"\n{'='*60}")
    print("Indirect Spillover Analysis")
    print(f"{'='*60}")

    direct_links = np.sum(A)
    
    analysis = {
        'direct_links': int(direct_links),
        'indirect_by_hop': {},
        'total_reachable': {}
    }
    
    cumulative_matrices = k_hops_results['cumulative_matrices']
    
    for k in range(2, k_hops_results['max_hops'] + 1):
        indirect_matrix = k_hops_results['k_hops_matrices'][k]
        indirect_links = np.sum(indirect_matrix)
        analysis['indirect_by_hop'][k] = int(indirect_links)
        
        cumulative = cumulative_matrices[k]
        total_reachable = np.sum(cumulative)
        analysis['total_reachable'][k] = int(total_reachable)
        
        print(f"\n{k}-hop indirect spillover:")
        print(f"  Direct links: {direct_links}")
        print(f"  {k}-hop indirect links: {indirect_links}")
        print(f"  Total reachable (up to {k}-hops): {total_reachable}")
        print(f"  Indirect spillover ratio: {indirect_links / (direct_links + 1e-10):.2%}")

    print(f"\n{'='*60}")
    print("Agent Indirect Influence Analysis")
    print(f"{'='*60}")
    
    for k in range(2, k_hops_results['max_hops'] + 1):
        indirect_matrix = k_hops_results['k_hops_matrices'][k]
        out_degree_indirect = np.sum(indirect_matrix, axis=0)
        in_degree_indirect = np.sum(indirect_matrix, axis=1)
        
        print(f"\n{k}-hop indirect influence:")
        print(f"  Out-degree (agents influenced indirectly):")
        for i in range(num_agents):
            print(f"    Agent {i}: {out_degree_indirect[i]}")
        print(f"  In-degree (agents that influence indirectly):")
        for i in range(num_agents):
            print(f"    Agent {i}: {in_degree_indirect[i]}")
    
    return analysis


def compute_temporal_spillover_operator(A_history, gamma_spatial=0.9, 
                                       gamma_temporal=0.95, K=3):
    """
    Args:
        A_history: List of influence matrices [A_early, A_middle, A_late, ...]
                   Each A_t is a binary matrix where A_t[j,i] = 1 if i influences j at time t
        gamma_spatial: Discount factor for K-hops (0.9 means 1-hop gets 0.9, 2-hop gets 0.81, etc.)
        gamma_temporal: Discount factor for time (0.95 means recent windows get higher weight)
        K: Maximum number of hops to consider
    
    Returns:
        S_temporal: Spillover operator matrix (n x n) with combined temporal-spatial weighting
        weights: Temporal weights used for each window
    """
    if len(A_history) == 0:
        raise ValueError("A_history must contain at least one influence matrix")
    
    n = A_history[0].shape[0]
    T = len(A_history)

    for i, A_t in enumerate(A_history):
        if A_t.shape != (n, n):
            raise ValueError(f"All matrices must have shape ({n}, {n}). Matrix {i} has shape {A_t.shape}")
    
    S = np.zeros((n, n))
    
    # Compute EWMA weights (most recent gets highest weight)
    # weights[t] = gamma_temporal^(T-1-t) / sum(weights)
    weights = np.array([gamma_temporal ** (T - 1 - t) for t in range(T)])
    weights /= weights.sum() 
    
    print(f"\n{'='*60}")
    print("Temporal Spillover Operator Computation")
    print(f"{'='*60}")
    print(f"Number of time windows: {T}")
    print(f"Spatial discount (gamma_spatial): {gamma_spatial}")
    print(f"Temporal discount (gamma_temporal): {gamma_temporal}")
    print(f"Max hops (K): {K}")
    print(f"\nTemporal weights (recent → old):")
    for t, w in enumerate(weights):
        print(f"  Window {t+1}: {w:.4f} ({w*100:.2f}%)")

    for t, A_t in enumerate(A_history):
        A_power = A_t.copy().astype(float)
        for k in range(1, K + 1):
            weight = weights[t] * (gamma_spatial ** k)
            S += weight * A_power
            if k < K:
                A_power = A_power @ A_t.astype(float)
    
    print(f"\nTemporal spillover operator computed:")
    print(f"  Matrix shape: {S.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(S)}")
    print(f"  Max value: {np.max(S):.4f}")
    print(f"  Min value: {np.min(S):.4f}")
    print(f"  Mean value: {np.mean(S):.4f}")
    
    print(f"\n{'='*60}")
    print("Temporal Spillover Operator Matrix S")
    print(f"{'='*60}")
    print("S[j,i] = weighted spillover from agent i to agent j")
    print("(combines temporal and spatial weighting)")
    print(f"\n{S}")
    print(f"\nMatrix formatted (rows = affected agents, cols = source agents):")
    print("     ", end="")
    for i in range(n):
        print(f"  Agent {i}", end="")
    print()
    for j in range(n):
        print(f"Agent {j}:", end="")
        for i in range(n):
            print(f"  {S[j, i]:6.4f}", end="")
        print()
    
    return S, weights


def compute_jacobian_finite_differences(trajectories, window_size=10):
    """
    Args:
        trajectories: Array of shape (T, num_agents) with time series data
        window_size: Window size for computing differences (default: 10)
    
    Returns:
        jacobian: Matrix of shape (num_agents, num_agents) with learning sensitivities
    """
    T, num_agents = trajectories.shape
    
    if T < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} data points, got {T}")
    
    print(f"\nComputing Jacobian using finite differences...")
    print(f"  Data points: {T}")
    print(f"  Window size: {window_size}")
    
    jacobian = np.zeros((num_agents, num_agents))
    differences = np.diff(trajectories, axis=0)
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                jacobian[j, i] = np.std(differences[:, i])
            else:
                if np.std(differences[:, i]) > 1e-10 and np.std(differences[:, j]) > 1e-10:
                    corr = np.corrcoef(differences[:, i], differences[:, j])[0, 1]
                    jacobian[j, i] = corr * np.std(differences[:, j])
                else:
                    jacobian[j, i] = 0.0

    max_val = np.max(np.abs(jacobian))
    if max_val > 0:
        jacobian = jacobian / max_val
    
    return jacobian


def compute_jacobian_correlation(trajectories):
    """
    Args:
        trajectories: Array of shape (T, num_agents) with time series data
    
    Returns:
        jacobian: Matrix of shape (num_agents, num_agents) with correlations
    """
    T, num_agents = trajectories.shape
    
    print(f"\nComputing Jacobian using correlation...")
    print(f"  Data points: {T}")
    
    jacobian = np.zeros((num_agents, num_agents))
    
    for i in range(num_agents):
        for j in range(num_agents):
            if np.std(trajectories[:, i]) > 1e-10 and np.std(trajectories[:, j]) > 1e-10:
                corr = np.corrcoef(trajectories[:, i], trajectories[:, j])[0, 1]
                jacobian[j, i] = corr
            else:
                jacobian[j, i] = 0.0
    
    return jacobian


def compute_tss(S, jacobian, normalize=True):
    """
    Args:
        S: Spillover operator (network structure + temporal weighting)
        jacobian: Learning sensitivity matrix
        normalize: If True, return as percentage
    
    Returns:
        tss: Total spillover strength
        tss_percent: TSS as percentage (if normalize=True)
    """
    num_agents = S.shape[0]
    
    if S.shape != jacobian.shape:
        raise ValueError(f"S shape {S.shape} must match jacobian shape {jacobian.shape}")
    
    print(f"\n{'='*60}")
    print("TSS Computation")
    print(f"{'='*60}")
    print(f"Spillover operator S shape: {S.shape}")
    print(f"Jacobian shape: {jacobian.shape}")

    tss_matrix = S * np.abs(jacobian)
    tss = np.sum(tss_matrix)
    
    print(f"\nTSS Matrix (S × |Jacobian|):")
    print(tss_matrix)
    print(f"\nRaw TSS: {tss:.4f}")
    
    if normalize:
        max_spillover = num_agents * (num_agents - 1)
        tss_percent = (tss / max_spillover) * 100 if max_spillover > 0 else 0
        
        print(f"Max possible spillover: {max_spillover}")
        print(f"TSS (normalized): {tss_percent:.2f}%")
        
        # Sanity check
        if 50 <= tss_percent <= 70:
            print("✓ TSS in expected range for typical MARL (50-70%)")
        elif tss_percent < 50:
            print("⚠ TSS below typical range - agents may be too independent")
        else:
            print("⚠ TSS above typical range - potential training instability")
        
        return tss, tss_percent
    
    return tss, None


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
    parser.add_argument("--k-hops", type=int, default=0,
                       help="Analyze k-hops propagation (0 = disabled, 1-5 = max hops to analyze)")
    parser.add_argument("--temporal-analysis", action="store_true",
                       help="Include temporal analysis with k-hops (requires k-hops > 0)")
    parser.add_argument("--time-windows", type=str, default=None,
                       help="Time window boundaries (comma-separated, e.g., '0,50000,100000,150000')")
    
    args = parser.parse_args()

    print(f"Loading trajectory data from: {args.trajectory_file}")
    data = load_trajectory_data(args.trajectory_file)

    print(f"\nPreparing trajectories using metric: {args.metric}")
    trajectories, timesteps = prepare_trajectories(data, metric=args.metric, max_lag=args.max_lag)

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

    k_hops_results = None
    indirect_analysis = None
    temporal_k_hops = None
    
    if args.k_hops > 0:
        k_hops_results = compute_k_hops_propagation(A, max_hops=args.k_hops)
        indirect_analysis = analyze_indirect_spillover(A, k_hops_results)

        if args.temporal_analysis:
            time_windows = None
            if args.time_windows:
                time_windows = [float(x.strip()) for x in args.time_windows.split(',')]
            
            temporal_k_hops = compute_k_hops_propagation_with_time(
                A, trajectories, timesteps, 
                max_hops=args.k_hops,
                time_windows=time_windows
            )
            
            temporal_spillover_operator = None
            temporal_weights = None
            if temporal_k_hops and 'temporal_windows' in temporal_k_hops:
                windows = temporal_k_hops['temporal_windows']
                if len(windows) > 1:
                    A_history = []
                    for window_idx in sorted(windows.keys()):
                        A_history.append(windows[window_idx]['influence_matrix'])
                    
                    # Compute temporal spillover operator
                    temporal_spillover_operator, temporal_weights = compute_temporal_spillover_operator(
                        A_history,
                        gamma_spatial=0.9,
                        gamma_temporal=0.95,
                        K=args.k_hops
                    )
                    
                    temporal_k_hops['temporal_spillover_operator'] = temporal_spillover_operator
                    temporal_k_hops['temporal_weights'] = temporal_weights
                    
                    print(f"\n{'='*60}")
                    print("Temporal Spillover Operator Summary")
                    print(f"{'='*60}")
                    print("Final weighted spillover matrix (combines all time windows):")
                    print(temporal_spillover_operator)
                    print(f"\nInterpretation:")
                    print(f"  - S[j,i] = weighted spillover from agent i to agent j")
                    
                    # Compute Jacobian and TSS
                    print(f"\n{'='*60}")
                    print("Computing Jacobian and TSS")
                    print(f"{'='*60}")
                    
                    # Method 1: Finite differences
                    try:
                        jacobian_fd = compute_jacobian_finite_differences(
                            trajectories, 
                            window_size=10
                        )
                        
                        # Method 2: Correlation (simple baseline)
                        jacobian_corr = compute_jacobian_correlation(trajectories)
                        
                        print(f"\n{'='*60}")
                        print("Jacobian Comparison")
                        print(f"{'='*60}")
                        print("\nFinite Differences Jacobian:")
                        print(jacobian_fd)
                        print("\nCorrelation Jacobian:")
                        print(jacobian_corr)
                        
                        # Compute TSS with both methods
                        print(f"\n{'='*60}")
                        print("TSS with Finite Differences Jacobian")
                        print(f"{'='*60}")
                        tss_fd, tss_fd_pct = compute_tss(temporal_spillover_operator, jacobian_fd, normalize=True)
                        
                        print(f"\n{'='*60}")
                        print("TSS with Correlation Jacobian")
                        print(f"{'='*60}")
                        tss_corr, tss_corr_pct = compute_tss(temporal_spillover_operator, jacobian_corr, normalize=True)
                        
                        # Add to results
                        temporal_k_hops['jacobian_finite_differences'] = jacobian_fd
                        temporal_k_hops['jacobian_correlation'] = jacobian_corr
                        temporal_k_hops['tss_finite_differences'] = tss_fd
                        temporal_k_hops['tss_finite_differences_percent'] = tss_fd_pct
                        temporal_k_hops['tss_correlation'] = tss_corr
                        temporal_k_hops['tss_correlation_percent'] = tss_corr_pct
                        
                    except Exception as e:
                        print(f"⚠ Warning: Could not compute Jacobian/TSS: {e}")
                        import traceback
                        traceback.print_exc()

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
            'alpha': args.alpha,
            'k_hops_results': k_hops_results,
            'indirect_analysis': indirect_analysis,
            'temporal_k_hops': temporal_k_hops
        }
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()

