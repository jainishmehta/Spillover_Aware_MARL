import numpy as np
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def load_trajectory_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
def prepare_trajectories(data, metric='agent_values', max_lag=5):
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


def test_stationarity(trajectories, agent_names=None):
    T, num_agents = trajectories.shape
    
    if agent_names is None:
        agent_names = [f"Agent {i}" for i in range(num_agents)]
    
    results = {
        'adf': {},
        'kpss': {},
        'is_stationary': {}
    }
    
    for i in range(num_agents):
        series = trajectories[:, i]
        
        # ADF Test (null: non-stationary)
        try:
            adf_result = adfuller(series, autolag='AIC')
            adf_stat, adf_pvalue = adf_result[0], adf_result[1]
            results['adf'][i] = {
                'statistic': adf_stat,
                'pvalue': adf_pvalue,
                'stationary': adf_pvalue < 0.05
            }
            print(f"\n{agent_names[i]} - ADF Test:")
            print(f"  Statistic: {adf_stat:.4f}")
            print(f"  p-value: {adf_pvalue:.4f}")
            if adf_pvalue < 0.05:
                print(f"  Stationary: Under 0.05 p-value")
            else:
                print(f"  Stationary: Over 0.05 p-value")
        except Exception as e:
            print(f"  ADF test failed: {e}")
            results['adf'][i] = {'stationary': False}
        
        # KPSS Test (null: stationary)
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
            results['kpss'][i] = {
                'statistic': kpss_stat,
                'pvalue': kpss_pvalue,
                'stationary': kpss_pvalue > 0.05
            }
            print(f"\n{agent_names[i]} - KPSS Test:")
            print(f"  Statistic: {kpss_stat:.4f}")
            print(f"  p-value: {kpss_pvalue:.4f}")
            if kpss_pvalue > 0.05:
                print(f"  Stationary: Over 0.05 p-value")
            else:
                print(f"  Stationary: Under 0.05 p-value")
        except Exception as e:
            print(f"  KPSS test failed: {e}")
            results['kpss'][i] = {'stationary': False}

        adf_stationary = results['adf'][i].get('stationary', False)
        kpss_stationary = results['kpss'][i].get('stationary', False)
        results['is_stationary'][i] = adf_stationary

    return results


def make_stationary(trajectories, method='diff'):
    if method == 'diff':
        stationary = np.diff(trajectories, axis=0)
    elif method == 'diff2':
        stationary = np.diff(np.diff(trajectories, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return stationary, trajectories


def estimate_influence_granger(trajectories, max_lag=5, alpha=0.1, method='fdr_bh', 
                                check_stationarity=True, make_stationary_if_needed=True, quiet=False):
    T, num_agents = trajectories.shape
    
    A = np.zeros((num_agents, num_agents), dtype=int)
    p_values = np.ones((num_agents, num_agents))
    test_statistics = np.zeros((num_agents, num_agents))
    
    if not quiet:
        print(f"Data shape: {trajectories.shape}")
        print(f"Max lag: {max_lag}")
        print(f"Significance level: {alpha}")
        print(f"FDR correction method: {method}\n")

    original_trajectories = trajectories.copy()
    if check_stationarity:
        if not quiet:
            stationarity_results = test_stationarity(trajectories)
        else:
            stationarity_results = {'is_stationary': {}}
            for i in range(trajectories.shape[1]):
                try:
                    adf_result = adfuller(trajectories[:, i], autolag='AIC')
                    stationarity_results['is_stationary'][i] = adf_result[1] < 0.05
                except:
                    stationarity_results['is_stationary'][i] = False
        
        non_stationary = [i for i, is_stat in stationarity_results['is_stationary'].items() if not is_stat]
        
        if non_stationary:
            if make_stationary_if_needed:
                if not quiet:
                    print("Proceeding with differencing to ensure valid statistical tests")
                trajectories, _ = make_stationary(trajectories, method='diff')
                T = trajectories.shape[0]
                
                if not quiet:
                    stationarity_results = test_stationarity(trajectories)
            else:
                if not quiet:
                    print(f"\nWarning: {len(non_stationary)} series appear non-stationary but differencing is DISABLED (--no-differencing flag set).")
    
    try:
        model = VAR(trajectories)
        lag_order = model.select_order(maxlags=max_lag)
        optimal_lag = lag_order.selected_orders.get('aic', max_lag)
        optimal_lag = min(optimal_lag, max_lag)
        if optimal_lag == 0:
            if not quiet:
                print(f"Optimal lag is 0, but Granger causality requires lag >= 1")
                print(f"Using minimum lag of 1 for Granger causality tests")
            optimal_lag = 1
        
        if not quiet:
            print(f"Optimal lag selected: {optimal_lag}")
        
        fitted_model = model.fit(optimal_lag)
        
        if not quiet:
            print(f"  - AIC: {fitted_model.aic:.4f}")
            print(f"  - BIC: {fitted_model.bic:.4f}")
            
            try:
                for i in range(num_agents):
                    residuals_i = fitted_model.resid[:, i]
                    if residuals_i.ndim > 1:
                        residuals_i = residuals_i.flatten()
                    ljung_box = acorr_ljungbox(residuals_i, lags=min(10, (T - optimal_lag) // 4), return_df=True)
                    lb_pvalue = ljung_box['lb_pvalue'].iloc[-1]
                    if lb_pvalue < 0.05:
                        print(f"    Agent {i}: p={lb_pvalue:.4f}, serial correlation detected")
                    else:
                        print(f"    Agent {i}: p={lb_pvalue:.4f} no serial correlation")
            except Exception as e:
                print(f"  - Residual diagnostics failed: {e}")
        
    except Exception as e:
        optimal_lag = max_lag
        fitted_model = model.fit(optimal_lag)
    if not quiet:
        print(f"\nTesting {num_agents * (num_agents - 1)} pairs for Granger causality")
    
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
                if not quiet:
                    print(f"Warning: Granger test failed for {i} -> {j}: {e}")
                p_values[j, i] = 1.0
                test_statistics[j, i] = 0.0
    if len(all_p_values) == 0:
        if not quiet:
            print(f"Influence matrix will be all zeros")
        return A, p_values, test_statistics
    if not quiet:
        print(f"\nRaw p-values (before FDR correction):")
        idx = 0
        for j in range(num_agents):
            for i in range(num_agents):
                if i != j:
                    print(f"  Agent {i} -> Agent {j}: p={all_p_values[idx]:.6f}")
                    idx += 1
        
        print(f"\nFDR correction ({method})")
    all_p_values = np.array(all_p_values)
    if len(all_p_values) == 0 or np.all(np.isnan(all_p_values)):
        return A, p_values, test_statistics
    
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
                if not quiet:
                    print(f"  Agent {i} -> Agent {j}: p={p_values[j,i]:.4f}, corrected_p={pvals_corrected[corrected_idx]:.4f}")
            
            corrected_idx += 1
    
    if not quiet:
        print(f"Influence Matrix (A[j,i] = 1 if i Granger-causes j):")
        print(A)
        print(f"\nTotal causal relationships found: {np.sum(A)}")
    
    return A, p_values, test_statistics


def compute_influence_statistics(A, A_ground_truth=None):
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

def compute_k_hops_propagation_with_time(A, trajectories, timesteps, max_hops=3, time_windows=None, 
                                          rolling_window_size=None, rolling_step_size=None, 
                                          num_temporal_windows=None, data=None):
    T, num_agents = trajectories.shape
    print(f"Temporal K-Hops Propagation Analysis")
    print(f"Data points: {T}")
    print(f"Timestep range: {timesteps[0] if len(timesteps) > 0 else 'N/A'} to {timesteps[-1] if len(timesteps) > 0 else 'N/A'}")
    
    results = {
        'global_k_hops': None,
        'temporal_windows': {},
        'rolling_analysis': {}
    }
    global_k_hops = compute_k_hops_propagation(A, max_hops=max_hops)
    results['global_k_hops'] = global_k_hops

    if time_windows is None:
        min_window_size = max((5 * num_agents * 2) + 10, 150)
        if num_temporal_windows is not None:
            target_num_windows = num_temporal_windows
        else:
            max_windows = max(1, int((T - min_window_size) / min_window_size))
            target_num_windows = min(3, max_windows)
        
        if len(timesteps) > 0:
            total_time = timesteps[-1] - timesteps[0]
            if target_num_windows == 0:
                target_num_windows = 1
            window_size = total_time / target_num_windows
            time_windows = [timesteps[0] + i * window_size for i in range(target_num_windows + 1)]
        else:
            if target_num_windows == 0:
                target_num_windows = 1
            window_size = T / target_num_windows
            time_windows = [int(i * window_size) for i in range(target_num_windows + 1)]
        
        print(f"Auto-generated {target_num_windows} time windows (minimum {min_window_size} points per window)")
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
        min_required = (5 * num_agents * 2) + 10
        if len(window_data) < min_required:
            print(f"\nWindow {window_idx + 1} ({window_start:.0f}-{window_end:.0f}):")
            print(f"  Insufficient data: {len(window_data)} points (minimum: {min_required})")
            continue
        data_indices = np.where(mask)[0]
        first_idx = data_indices[0] if len(data_indices) > 0 else None
        last_idx = data_indices[-1] if len(data_indices) > 0 else None
        episode_info = ""
        if data is not None:
            episode_timesteps = data.get('episode_timesteps', None)
            if episode_timesteps is not None and len(episode_timesteps) > 0:
                if len(timesteps) > 0:
                    window_timesteps = timesteps[mask]
                    if len(window_timesteps) > 0:
                        min_ts = window_timesteps[0]
                        max_ts = window_timesteps[-1]
                        episode_mask = (episode_timesteps >= min_ts) & (episode_timesteps <= max_ts)
                        episode_nums = np.where(episode_mask)[0]
                        if len(episode_nums) > 0:
                            episode_info = f"  Episodes: {episode_nums[0]} to {episode_nums[-1]} (approx {len(episode_nums)} episodes)"
        
        print(f"\nWindow {window_idx + 1} ({window_start:.0f}-{window_end:.0f}):")
        if len(timesteps) > 0:
            window_timesteps = timesteps[mask]
            if len(window_timesteps) > 0:
                print(f"  Timestep range: {window_timesteps[0]:.0f} to {window_timesteps[-1]:.0f}")
        if episode_info:
            print(episode_info)

        try:
            adaptive_max_lag = 5
            for test_lag in range(5, 0, -1):
                required_obs = (test_lag * num_agents * 2) + 10
                if len(window_data) >= required_obs:
                    adaptive_max_lag = test_lag
                    break
            adaptive_max_lag = max(1, adaptive_max_lag)
            
            print(f"  Using max_lag: {adaptive_max_lag} (based on {len(window_data)} observations)")
            
            A_window, _, _ = estimate_influence_granger(
                window_data, 
                max_lag=adaptive_max_lag,
                alpha=0.1
            )
            k_hops_window = compute_k_hops_propagation(A_window, max_hops=max_hops)
            window_info = {
                'time_range': (window_start, window_end),
                'data_indices': (first_idx, last_idx) if first_idx is not None else None,
                'influence_matrix': A_window,
                'k_hops': k_hops_window,
                'num_data_points': len(window_data),
                'data_mask': mask
            }
            if len(timesteps) > 0:
                window_timesteps = timesteps[mask]
                if len(window_timesteps) > 0:
                    window_info['timestep_range'] = (float(window_timesteps[0]), float(window_timesteps[-1]))

            if data is not None:
                episode_timesteps = data.get('episode_timesteps', None)
                if episode_timesteps is not None and len(timesteps) > 0:
                    window_timesteps = timesteps[mask]
                    if len(window_timesteps) > 0:
                        min_ts = window_timesteps[0]
                        max_ts = window_timesteps[-1]
                        episode_mask = (episode_timesteps >= min_ts) & (episode_timesteps <= max_ts)
                        episode_nums = np.where(episode_mask)[0]
                        if len(episode_nums) > 0:
                            window_info['episode_range'] = (int(episode_nums[0]), int(episode_nums[-1]))
                            window_info['num_episodes'] = len(episode_nums)
            
            results['temporal_windows'][window_idx] = window_info
            
            print(f"  Direct links: {np.sum(A_window)}")
            for k in range(2, max_hops + 1):
                indirect = k_hops_window['k_hops_matrices'][k]
                print(f"  {k}-hop indirect links: {np.sum(indirect)}")
        
        except Exception as e:
            print(f"  Error analyzing window: {e}")
            results['temporal_windows'][window_idx] = {
                'time_range': (window_start, window_end),
                'influence_matrix': np.zeros((num_agents, num_agents), dtype=int),
                'num_data_points': len(window_data),
                'error': str(e)
            }
            continue

    print(f"\n{'='*60}")
    print("Rolling Window Analysis")
    print(f"{'='*60}")
    min_window_size = (3 * num_agents * 2) + 10
    
    if rolling_window_size is not None:
        if rolling_window_size < min_window_size:
            print(f"Using minimum window size: {min_window_size}")
            window_size = min_window_size
        elif rolling_window_size > T:
            print(f"Warning: Specified window size ({rolling_window_size}) exceeds data length ({T})")
            window_size = T
        else:
            window_size = rolling_window_size
    else:
        window_size = max(min_window_size, min(150, T // 2))
    
    if rolling_step_size is not None:
        if rolling_step_size < 1:
            step_size = 1
        elif rolling_step_size >= window_size:
            step_size = max(1, window_size // 2)
        else:
            step_size = rolling_step_size
    else:
        step_size = max(1, window_size // 2)

    rolling_results = []
    
    for start_idx in range(0, T - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = trajectories[start_idx:end_idx]
        
        if len(timesteps) > 0:
            window_times = (timesteps[start_idx], timesteps[end_idx-1])
        else:
            window_times = (start_idx, end_idx-1)

        min_required = (3 * num_agents * 2) + 10
        if len(window_data) < min_required:
            continue
        
        try:
            adaptive_max_lag = 3
            for test_lag in range(3, 0, -1):
                required_obs = (test_lag * num_agents * 2) + 10
                if len(window_data) >= required_obs:
                    adaptive_max_lag = test_lag
                    break
            adaptive_max_lag = max(1, adaptive_max_lag)
            
            A_rolling, _, _ = estimate_influence_granger(
                window_data,
                max_lag=adaptive_max_lag,
                alpha=0.1
            )
            
            direct_links = np.sum(A_rolling)
            rolling_results.append({
                'time_range': window_times,
                'data_range': (start_idx, end_idx),
                'influence_matrix': A_rolling,
                'direct_links': int(direct_links),
                'max_lag_used': adaptive_max_lag
            })
        
        except Exception as e:
            continue
    
    results['rolling_analysis'] = {
        'window_size': window_size,
        'step_size': step_size,
        'results': rolling_results
    }

    if len(rolling_results) > 0:
        direct_links_over_time = [r['direct_links'] for r in rolling_results]
        print(f"  Mean direct links: {np.mean(direct_links_over_time):.2f}")
        print(f"  Std direct links: {np.std(direct_links_over_time):.2f}")
        print(f"  Range: [{np.min(direct_links_over_time)}, {np.max(direct_links_over_time)}]")
    
    return results


def compute_k_hops_propagation(A, max_hops=3):
    num_agents = A.shape[0]
    print(f"K-Hops Propagation Analysis (up to {max_hops} hops)")
    
    k_hops_matrices = {}
    k_hops_paths = {}

    k_hops_matrices[1] = A.copy()
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
                print(f"    Agent {i} → Agent {j} in {k} hops)")
    
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
    if len(A_history) == 0:
        raise ValueError("A_history must contain at least one influence matrix")
    
    n = A_history[0].shape[0]
    T = len(A_history)

    for i, A_t in enumerate(A_history):
        if A_t.shape != (n, n):
            raise ValueError(f"All matrices must have shape ({n}, {n}). Matrix {i} has shape {A_t.shape}")
    
    S = np.zeros((n, n))

    weights = np.array([gamma_temporal ** (T - 1 - t) for t in range(T)])
    weights /= weights.sum() 
    
    print(f"\n{'='*60}")
    print("Temporal Spillover Operator Computation")
    print(f"{'='*60}")
    print(f"Number of time windows: {T}")
    print(f"Spatial discount (gamma_spatial): {gamma_spatial}")
    print(f"Temporal discount (gamma_temporal): {gamma_temporal}")
    print(f"Max hops (K): {K}")
    for t, w in enumerate(weights):
        print(f"  Window {t+1}: {w:.4f} ({w*100:.2f}%)")

    for t, A_t in enumerate(A_history):
        A_power = A_t.copy().astype(float)
        for k in range(1, K + 1):
            weight = weights[t] * (gamma_spatial ** k)
            S += weight * A_power
            if k < K:
                A_power = A_power @ A_t.astype(float)

    print(f"  Matrix shape: {S.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(S)}")
    print(f"  Max value: {np.max(S):.4f}")
    print(f"  Min value: {np.min(S):.4f}")
    print(f"  Mean value: {np.mean(S):.4f}")
    
    print(f"\n{'='*60}")
    print("Temporal Spillover Operator Matrix S")
    print(f"{'='*60}")
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
    T, num_agents = trajectories.shape
    
    if T < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} data points, got {T}")

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


def analyze_tss_factors(S, jacobian, A_history=None):
    num_agents = S.shape[0]
    
    print(f"\n{'='*60}")
    print("TSS Factor Analysis")
    print(f"{'='*60}")

    S_nonzero = np.count_nonzero(S)
    S_max = np.max(S)
    S_mean = np.mean(S[S > 0]) if S_nonzero > 0 else 0
    S_total = np.sum(S)
    
    print(f"\nSpillover Operator (S) Analysis:")
    print(f"  Total sum: {S_total:.4f}")
    print(f"  Non-zero entries: {S_nonzero} / {num_agents * num_agents}")
    print(f"  Max value: {S_max:.4f}")
    print(f"  Mean (non-zero): {S_mean:.4f}")
    
    jacobian_abs = np.abs(jacobian)
    jacobian_nonzero = np.count_nonzero(jacobian_abs)
    jacobian_max = np.max(jacobian_abs)
    jacobian_mean = np.mean(jacobian_abs[jacobian_abs > 0]) if jacobian_nonzero > 0 else 0
    jacobian_total = np.sum(jacobian_abs)
    
    print(f"\nJacobian (|J|) Analysis:")
    print(f"  Total sum: {jacobian_total:.4f}")
    print(f"  Non-zero entries: {jacobian_nonzero} / {num_agents * num_agents}")
    print(f"  Max value: {jacobian_max:.4f}")
    print(f"  Mean (non-zero): {jacobian_mean:.4f}")

    tss_matrix = S * jacobian_abs
    tss_total = np.sum(tss_matrix)
    tss_max = np.max(tss_matrix)
    tss_mean = np.mean(tss_matrix[tss_matrix > 0]) if np.count_nonzero(tss_matrix) > 0 else 0
    
    print(f"\nTSS Matrix (S × |J|) Analysis:")
    print(f"  Total TSS: {tss_total:.4f}")
    print(f"  Max value: {tss_max:.4f}")
    print(f"  Mean (non-zero): {tss_mean:.4f}")

    recommendations = []
    
    if S_total < num_agents * 0.5:
        recommendations.append("LOW SPILLOVER OPERATOR: Increase causal relationships detected, - Lower alpha (e.g., --alpha 0.01) to detect more causal links, - Increase k-hops (e.g., --k-hops 5) to include more indirect paths, Increase gamma_spatial (e.g., --gamma-spatial 0.95) to weight indirect paths more")
    
    if jacobian_total < num_agents * 0.5:
        recommendations.append("LOW JACOBIAN: Agents may not be strongly correlated, - Check if agents are actually learning together,- Consider using different metric (--metric policy_params or episode_rewards), - Ensure sufficient training data for stable correlations")
    
    if S_nonzero < num_agents * (num_agents - 1) * 0.3:
        recommendations.append("SPARSE CAUSAL NETWORK: Few causal relationships detected")
        recommendations.append("  - Use --no-differencing, - Increase temporal windows (--num-temporal-windows 4-5), - Check stationarity - may need different preprocessing")
    
    if tss_total < num_agents * (num_agents - 1) * 0.1:
        recommendations.append("LOW TSS OVERALL: Both S and Jacobian may need improvement, - Increase gamma_temporal (e.g., --gamma-temporal 0.98) for recent windows, - Ensure agents have meaningful interactions in the environment")
    
    if recommendations:
        print(f"\n{'='*60}")
        print("Recommendations to Increase TSS:")
        print(f"{'='*60}")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print(f"\n TSS factors appear well-balanced")
    
    return {
        'S_total': S_total,
        'S_nonzero': S_nonzero,
        'jacobian_total': jacobian_total,
        'jacobian_nonzero': jacobian_nonzero,
        'tss_total': tss_total,
        'recommendations': recommendations
    }


def compute_tss(S, jacobian, normalize=True):
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
    print(tss_matrix)
    print(f"\nRaw TSS: {tss:.4f}")
    
    if normalize:
        max_spillover = num_agents * (num_agents - 1)
        tss_percent = (tss / max_spillover) * 100 if max_spillover > 0 else 0
        
        print(f"Max possible spillover: {max_spillover}")
        print(f"TSS (normalized): {tss_percent:.2f}%")
        if 50 <= tss_percent <= 70:
            print("TSS in expected range for typical MARL (50-70%)")
        elif tss_percent < 50:
            print("TSS below typical range, agents too independent")
        else:
            print("TSS above typical range, training instability")
        
        return tss, tss_percent
    
    return tss, None


def test_multiple_window_sizes(trajectories, timesteps, window_sizes=[100, 125, 175, 200], 
                                max_lag=5, alpha=0.1, make_stationary_if_needed=True):
    T, num_agents = trajectories.shape
    print(f"\n{'='*60}")
    print("Testing Multiple Window Sizes")
    print(f"{'='*60}")
    print(f"Testing window sizes: {window_sizes}")
    print(f"Total data points: {T}")
    
    results = {}
    
    for window_size in window_sizes:
        if window_size > T:
            print(f"\nSkipping window size {window_size}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing window size: {window_size}")
        print(f"{'='*60}")
        
        step_size = max(1, window_size // 2)  # 50% overlap
        num_windows = max(1, (T - window_size) // step_size + 1)
        
        print(f"Step size: {step_size}")
        print(f"Number of windows: {num_windows}")
        
        window_results = []
        causal_links_per_window = []
        
        for start_idx in range(0, T - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = trajectories[start_idx:end_idx]

            min_required = (max_lag * num_agents * 2) + 10
            if len(window_data) < min_required:
                continue
            
            try:
                adaptive_max_lag = max_lag
                for test_lag in range(max_lag, 0, -1):
                    required_obs = (test_lag * num_agents * 2) + 10
                    if len(window_data) >= required_obs:
                        adaptive_max_lag = test_lag
                        break
                adaptive_max_lag = max(1, adaptive_max_lag)
                
                A_window, p_values_window, _ = estimate_influence_granger(
                    window_data,
                    max_lag=adaptive_max_lag,
                    alpha=alpha,
                    check_stationarity=True,
                    make_stationary_if_needed=make_stationary_if_needed,
                    quiet=True
                )
                
                num_links = np.sum(A_window)
                causal_links_per_window.append(num_links)
                
                window_results.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_links': int(num_links),
                    'influence_matrix': A_window,
                    'p_values': p_values_window
                })
                
            except Exception as e:
                continue
        
        if len(causal_links_per_window) > 0:
            results[window_size] = {
                'window_results': window_results,
                'mean_links': np.mean(causal_links_per_window),
                'std_links': np.std(causal_links_per_window),
                'min_links': np.min(causal_links_per_window),
                'max_links': np.max(causal_links_per_window),
                'num_windows': len(window_results),
                'stability': np.std(causal_links_per_window) / (np.mean(causal_links_per_window) + 1e-10)  # Coefficient of variation
            }
            
            print(f"\nSummary for window size {window_size}:")
            print(f"  Mean causal links: {results[window_size]['mean_links']:.2f}")
            print(f"  Std causal links: {results[window_size]['std_links']:.2f}")
            print(f"  Range: [{results[window_size]['min_links']}, {results[window_size]['max_links']}]")
            print(f"  Stability (CV): {results[window_size]['stability']:.3f} (lower is better)")
            print(f"  Windows analyzed: {results[window_size]['num_windows']}")
        else:
            print(f"\nNo valid windows for size {window_size}")
    
    return results


def plot_causal_vs_training_curves(temporal_results, data, save_path=None):
    if 'temporal_windows' not in temporal_results or len(temporal_results['temporal_windows']) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    windows = temporal_results['temporal_windows']
    window_indices = sorted(windows.keys())

    ax1 = axes[0, 0]
    window_midpoints = []
    num_links = []
    for idx in window_indices:
        window_info = windows[idx]
        if 'timestep_range' in window_info:
            midpoint = (window_info['timestep_range'][0] + window_info['timestep_range'][1]) / 2
        elif 'data_indices' in window_info and window_info['data_indices']:
            midpoint = (window_info['data_indices'][0] + window_info['data_indices'][1]) / 2
        else:
            midpoint = idx
        window_midpoints.append(midpoint)
        num_links.append(np.sum(window_info['influence_matrix']))
    
    ax1.plot(window_midpoints, num_links, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Timestep (midpoint of window)', fontsize=12)
    ax1.set_ylabel('Number of Causal Links', fontsize=12)
    ax1.set_title('Causal Relationships Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    num_agents = windows[window_indices[0]]['influence_matrix'].shape[0]
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                links = []
                for idx in window_indices:
                    A = windows[idx]['influence_matrix']
                    links.append(A[j, i])
                ax2.plot(window_midpoints, links, 'o-', label=f'Agent {i} → Agent {j}', 
                        linewidth=2, markersize=6)
    ax2.set_xlabel('Timestep (midpoint of window)', fontsize=12)
    ax2.set_ylabel('Causal Link (1=present, 0=absent)', fontsize=12)
    ax2.set_title('Individual Causal Relationships', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    ax3 = axes[1, 0]
    if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
        episode_timesteps = data.get('episode_timesteps', None)
        for agent_idx in range(num_agents):
            if agent_idx in data['episode_rewards']:
                rewards = data['episode_rewards'][agent_idx]
                if episode_timesteps is not None and len(episode_timesteps) == len(rewards):
                    ax3.plot(episode_timesteps, rewards, alpha=0.6, label=f'Agent {agent_idx}', linewidth=1.5)
                else:
                    ax3.plot(rewards, alpha=0.6, label=f'Agent {agent_idx}', linewidth=1.5)
        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('Episode Reward', fontsize=12)
        ax3.set_title('Training Curves: Episode Rewards', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for idx in window_indices:
            window_info = windows[idx]
            if 'timestep_range' in window_info:
                ax3.axvline(window_info['timestep_range'][0], color='gray', linestyle='--', alpha=0.5)
                ax3.axvline(window_info['timestep_range'][1], color='gray', linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'Episode rewards not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Curves: Episode Rewards', fontsize=14, fontweight='bold')

    ax4 = axes[1, 1]
    if 'agent_values' in data:
        timesteps_all = data.get('timesteps', np.arange(len(data['agent_values'][0])))
        for agent_idx in range(num_agents):
            if agent_idx in data['agent_values']:
                values = data['agent_values'][agent_idx]
                ax4.plot(timesteps_all[:len(values)], values, alpha=0.6, 
                        label=f'Agent {agent_idx}', linewidth=1.5)

        for idx in window_indices:
            window_info = windows[idx]
            if 'timestep_range' in window_info:
                ax4.axvline(window_info['timestep_range'][0], color='red', linestyle='--', alpha=0.5, linewidth=2)
                ax4.axvline(window_info['timestep_range'][1], color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        ax4.set_xlabel('Timestep', fontsize=12)
        ax4.set_ylabel('Agent Value', fontsize=12)
        ax4.set_title('Agent Values with Window Boundaries', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Agent values not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Agent Values', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining curve comparison plot saved to: {save_path}")
    else:
        plt.show()


def validate_reversal_pattern(temporal_results, min_consistency=2):
    if 'temporal_windows' not in temporal_results or len(temporal_results['temporal_windows']) == 0:
        return {'error': 'No temporal window results available'}
    
    windows = temporal_results['temporal_windows']
    window_indices = sorted(windows.keys())
    
    if len(window_indices) < 2:
        return {'error': 'Need at least 2 windows for validation'}
    
    print(f"\n{'='*60}")
    print("Reversal Pattern Validation")
    print(f"{'='*60}")
    
    num_agents = windows[window_indices[0]]['influence_matrix'].shape[0]

    patterns_over_time = []
    for idx in window_indices:
        A = windows[idx]['influence_matrix']
        out_degree = np.sum(A, axis=0)
        in_degree = np.sum(A, axis=1)
        
        patterns_over_time.append({
            'out_degree': out_degree.copy(),
            'in_degree': in_degree.copy(),
            'total_links': np.sum(A),
            'dominant_influencer': np.argmax(out_degree) if np.max(out_degree) > 0 else None,
            'most_influenced': np.argmax(in_degree) if np.max(in_degree) > 0 else None
        })

    validation_report = {
        'patterns': patterns_over_time,
        'reversals_detected': [],
        'is_valid': True,
        'warnings': []
    }
    dominant_influencers = [p['dominant_influencer'] for p in patterns_over_time if p['dominant_influencer'] is not None]
    if len(dominant_influencers) >= min_consistency:
        for i in range(len(dominant_influencers) - min_consistency + 1):
            window_group = dominant_influencers[i:i+min_consistency]
            if len(set(window_group)) == 1:
                if i + min_consistency < len(dominant_influencers):
                    next_group = dominant_influencers[i+min_consistency:i+2*min_consistency]
                    if len(next_group) == min_consistency and len(set(next_group)) == 1:
                        if window_group[0] != next_group[0]:
                            validation_report['reversals_detected'].append({
                                'type': 'dominant_influencer',
                                'from': window_group[0],
                                'to': next_group[0],
                                'window_range': (i, i + 2*min_consistency - 1)
                            })

    total_links = [p['total_links'] for p in patterns_over_time]
    links_std = np.std(total_links)
    links_mean = np.mean(total_links)
    if links_mean > 0:
        cv = links_std / links_mean

    print(f"\nPattern Analysis:")
    for i, pattern in enumerate(patterns_over_time):
        print(f"  Window {i+1}:")
        print(f"    Total links: {pattern['total_links']}")
        print(f"    Out-degree: {pattern['out_degree']}")
        print(f"    In-degree: {pattern['in_degree']}")
        if pattern['dominant_influencer'] is not None:
            print(f"    Dominant influencer: Agent {pattern['dominant_influencer']}")
        if pattern['most_influenced'] is not None:
            print(f"    Most influenced: Agent {pattern['most_influenced']}")
    
    if validation_report['reversals_detected']:
        print(f"\nReversals Detected:")
        for rev in validation_report['reversals_detected']:
            print(f"  {rev['type']}: Agent {rev['from']} → Agent {rev['to']} "
                  f"(windows {rev['window_range'][0]+1} to {rev['window_range'][1]+1})")
    else:
        print(f"\nNo clear reversals detected (need {min_consistency} consecutive windows)")
    
    if validation_report['warnings']:
        print(f"\nWarnings:")
        validation_report['is_valid'] = False
    else:
        print(f"\n Pattern appears statistically consistent")    
    return validation_report


def visualize_influence_matrix(A, p_values=None, save_path=None):
    
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
    parser.add_argument("--no-differencing", action="store_true",
                       help="Disable automatic differencing for non-stationary data")
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
    parser.add_argument("--max-timestep", type=float, default=None,
                       help="Maximum timestep to include in analysis (filters early training data)")
    parser.add_argument("--min-timestep", type=float, default=None,
                       help="Minimum timestep to include in analysis (filters late training data)")
    parser.add_argument("--rolling-window-size", type=int, default=None,
                       help="Size of rolling windows for temporal analysis (default: adaptive based on data length)")
    parser.add_argument("--rolling-step-size", type=int, default=None,
                       help="Step size for rolling windows (default: 50%% of window size for overlap)")
    parser.add_argument("--num-temporal-windows", type=int, default=None,
                       help="Number of temporal windows to analyze (default: adaptive, max 3)")
    parser.add_argument("--test-window-sizes", action="store_true",
                       help="Test multiple window sizes (100, 125, 175, 200) to find minimum reliable detection threshold")
    parser.add_argument("--plot-training-curves", action="store_true",
                       help="Plot causal relationships compared to training curves (rewards, losses)")
    parser.add_argument("--validate-reversal", action="store_true",
                       help="Validate reversal patterns for statistical consistency")
    parser.add_argument("--gamma-spatial", type=float, default=0.9,
                       help="Spatial discount factor for k-hops (0.0-1.0, higher = more weight on indirect paths, default: 0.9)")
    parser.add_argument("--gamma-temporal", type=float, default=0.95,
                       help="Temporal discount factor (0.0-1.0, higher = more weight on recent windows, default: 0.95)")
    
    args = parser.parse_args()

    print(f"Loading trajectory data from: {args.trajectory_file}")
    data = load_trajectory_data(args.trajectory_file)

    print(f"\nPreparing trajectories using metric: {args.metric}")
    trajectories, timesteps = prepare_trajectories(data, metric=args.metric, max_lag=args.max_lag)

    if args.min_timestep is not None or args.max_timestep is not None:
        if len(timesteps) == 0:
            print("Warning: No timestep information available, cannot filter by timestep")
        else:
            mask = np.ones(len(timesteps), dtype=bool)
            if args.min_timestep is not None:
                mask = mask & (timesteps >= args.min_timestep)
                print(f"Filtering: keeping timesteps >= {args.min_timestep}")
            if args.max_timestep is not None:
                mask = mask & (timesteps <= args.max_timestep)
                print(f"Filtering: keeping timesteps <= {args.max_timestep}")
            
            trajectories = trajectories[mask]
            timesteps = timesteps[mask]
            print(f"Filtered data: {len(trajectories)} points (from {len(data.get('timesteps', []))} original)")
            
            if len(trajectories) < args.max_lag + 10:
                raise ValueError(f"After filtering, only {len(trajectories)} points remain, need at least {args.max_lag + 10}")

    A, p_values, test_statistics = estimate_influence_granger(
        trajectories, 
        max_lag=args.max_lag, 
        alpha=args.alpha,
        make_stationary_if_needed=not args.no_differencing
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
                time_windows=time_windows,
                rolling_window_size=args.rolling_window_size,
                rolling_step_size=args.rolling_step_size,
                num_temporal_windows=args.num_temporal_windows,
                data=data
            )
            
            temporal_spillover_operator = None
            temporal_weights = None
            if temporal_k_hops and 'temporal_windows' in temporal_k_hops:
                windows = temporal_k_hops['temporal_windows']
                if len(windows) > 1:
                    A_history = []
                    for window_idx in sorted(windows.keys()):
                        A_history.append(windows[window_idx]['influence_matrix'])
                    
                    temporal_spillover_operator, temporal_weights = compute_temporal_spillover_operator(
                        A_history,
                        gamma_spatial=args.gamma_spatial,
                        gamma_temporal=args.gamma_temporal,
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
                    
                    print(f"\n{'='*60}")
                    print("Computing Jacobian and TSS")
                    print(f"{'='*60}")
                    
                    try:
                        jacobian_fd = compute_jacobian_finite_differences(
                            trajectories, 
                            window_size=10
                        )
                        
                        jacobian_corr = compute_jacobian_correlation(trajectories)
                        
                        print(f"\n{'='*60}")
                        print("Jacobian Comparison")
                        print(f"{'='*60}")
                        print("\nFinite Differences Jacobian:")
                        print(jacobian_fd)
                        print("\nCorrelation Jacobian:")
                        print(jacobian_corr)
                        print(f"\n{'='*60}")
                        print("TSS with Finite Differences Jacobian")
                        print(f"{'='*60}")
                        tss_fd, tss_fd_pct = compute_tss(temporal_spillover_operator, jacobian_fd, normalize=True)

                        tss_analysis_fd = analyze_tss_factors(temporal_spillover_operator, jacobian_fd, A_history)
                        
                        print(f"\n{'='*60}")
                        print("TSS with Correlation Jacobian")
                        print(f"{'='*60}")
                        tss_corr, tss_corr_pct = compute_tss(temporal_spillover_operator, jacobian_corr, normalize=True)
                        tss_analysis_corr = analyze_tss_factors(temporal_spillover_operator, jacobian_corr, A_history)

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

            if args.test_window_sizes:
                print(f"\n{'='*60}")
                print("Testing Multiple Window Sizes")
                print(f"{'='*60}")
                window_size_results = test_multiple_window_sizes(
                    trajectories, timesteps,
                    window_sizes=[100, 125, 175, 200],
                    max_lag=args.max_lag,
                    alpha=args.alpha,
                    make_stationary_if_needed=not args.no_differencing
                )
                if temporal_k_hops:
                    temporal_k_hops['window_size_comparison'] = window_size_results

            if args.validate_reversal and temporal_k_hops:
                validation_report = validate_reversal_pattern(temporal_k_hops, min_consistency=2)
                if temporal_k_hops:
                    temporal_k_hops['validation_report'] = validation_report

            if args.plot_training_curves and temporal_k_hops:
                plot_save_path = None
                if args.save_results:
                    plot_save_path = args.save_results.replace('.pkl', '_training_curves.png')
                plot_causal_vs_training_curves(temporal_k_hops, data, save_path=plot_save_path)

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

