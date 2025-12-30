import numpy as np
import pickle
import argparse
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


def test_stationarity(trajectories, agent_names=None):
    T, num_agents = trajectories.shape
    
    if agent_names is None:
        agent_names = [f"Agent {i}" for i in range(num_agents)]
    
    print("Stationarity Tests")
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
        print(f"Applied first differencing: {trajectories.shape} -> {stationary.shape}")
    elif method == 'diff2':
        stationary = np.diff(np.diff(trajectories, axis=0), axis=0)
        print(f"Applied second differencing: {trajectories.shape} -> {stationary.shape}")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return stationary, trajectories


def estimate_influence_granger(trajectories, max_lag=5, alpha=0.05, check_stationarity=True, make_stationary_if_needed=True):
    T, num_agents = trajectories.shape
    A = np.zeros((num_agents, num_agents), dtype=int)

    print(f"Data shape: {trajectories.shape}, Max lag: {max_lag}")

    original_trajectories = trajectories.copy()
    if check_stationarity:
        stationarity_results = test_stationarity(trajectories)
        
        non_stationary = [i for i, is_stat in stationarity_results['is_stationary'].items() if not is_stat]
        
        if non_stationary and make_stationary_if_needed:
            trajectories, _ = make_stationary(trajectories, method='diff')
            T = trajectories.shape[0] 
            stationarity_results = test_stationarity(trajectories)
        elif non_stationary:
            print("Results may be unreliable due to non-stationarity.")
    
    min_required = max_lag * num_agents * 2 
    if T < min_required:
        print(f"\nWarning: Only {T} observations. Recommended: at least {min_required} for reliable VAR({max_lag})")

    min_required = max_lag * num_agents * 2
    if T < min_required:
        print(f"Warning: Only {T} observations. Recommended: at least {min_required} for reliable VAR({max_lag})")

    model = VAR(trajectories)

    lag_order = model.select_order(maxlags=max_lag)
    optimal_lag = lag_order.selected_orders.get('aic', max_lag)
    optimal_lag = min(optimal_lag, max_lag)

    fitted_model = model.fit(optimal_lag)

    print("VAR Model Diagnostics")
    print(f"  - AIC: {fitted_model.aic:.4f}")
    print(f"  - BIC: {fitted_model.bic:.4f}")
    print(f"  - Effective observations: {T - optimal_lag}")
    print(f"  - Parameters per equation: {optimal_lag * num_agents}")
    print(f"  - Observations per parameter: {(T - optimal_lag) / (optimal_lag * num_agents):.2f}")
    try:
        for i in range(num_agents):
            residuals_i = fitted_model.resid[:, i]
            try:
                residuals_i = fitted_model.resid[:, i]
                if residuals_i.ndim > 1:
                    residuals_i = residuals_i.flatten()
                
                ljung_box = acorr_ljungbox(residuals_i, lags=min(10, (T - optimal_lag) // 4), return_df=True)
                lb_pvalue = ljung_box['lb_pvalue'].iloc[-1]
            except Exception as e:
                continue
            if lb_pvalue < 0.05:
                print(f"    Agent {i}, serial correlation detected")
            else:
                print(f"    Agent {i}, no serial correlation)")
    except Exception as e:
        print(f"  - Ljung-Box test failed: {e}")

    print(f"\nTesting {num_agents * (num_agents - 1)} pairs for Granger causality...")
    
    p_values = []
    test_results = []
    
    for j in range(num_agents):
        for i in range(num_agents):
            if i == j:
                continue 
            
            try:
                test_result = fitted_model.test_causality(j, i, kind='f')
                p_val = test_result.pvalue
                f_stat = test_result.test_statistic
                
                p_values.append(p_val)
                test_results.append((j, i, p_val, f_stat))
                
            except Exception as e:
                p_values.append(1.0)
                test_results.append((j, i, 1.0, 0.0))

    print(f"\nFDR correction (Benjamini-Hochberg) with alpha={alpha}")
    p_values = np.array(p_values)
    rejected, pvals_corrected, _, _ = multipletests(
        p_values, alpha=alpha, method='fdr_bh'
    )

    idx = 0
    for j in range(num_agents):
        for i in range(num_agents):
            if i == j:
                continue
            if rejected[idx]:
                A[j, i] = 1
                print(f" Agent {i} -> Agent {j}: p={p_values[idx]:.4f}, "
                      f"corrected_p={pvals_corrected[idx]:.4f}")
            else:
                if p_values[idx] < 0.1:
                    print(f"  - Agent {i} -> Agent {j}: p={p_values[idx]:.4f}, "
                          f"corrected_p={pvals_corrected[idx]:.4f}")
            idx += 1
    
    print(f"\nInfluence Matrix A (A[j,i] = 1 if i Granger-causes j):")
    print(A)
    print(f"\nTotal causal relationships: {np.sum(A)}")
    
    return A

def validate_influence_matrix(A_hat, A_true):
    num_agents = A_hat.shape[0]

    mask = ~np.eye(num_agents, dtype=bool)
    A_hat_flat = A_hat[mask]
    A_true_flat = A_true[mask]

    tp = np.sum((A_hat_flat == 1) & (A_true_flat == 1))
    fp = np.sum((A_hat_flat == 1) & (A_true_flat == 0))
    fn = np.sum((A_hat_flat == 0) & (A_true_flat == 1))
    tn = np.sum((A_hat_flat == 0) & (A_true_flat == 0)) 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    return metrics

def test_time_windows(trajectories, max_lag=5, alpha=0.05, window_sizes=None):
    T, num_agents = trajectories.shape
    
    if window_sizes is None:
        window_sizes = [int(T * 0.5), int(T * 0.75), T]
    
    results = {}
    
    for window_size in window_sizes:
        if window_size < max_lag * 2:
            continue

        window_data = trajectories[-window_size:]
        
        try:
            A_window = estimate_influence_granger(
                window_data, max_lag=max_lag, alpha=alpha,
                check_stationarity=False, make_stationary_if_needed=False
            )
            results[window_size] = A_window
            print(f"  Causal links found: {np.sum(A_window)}")
        except Exception as e:
            print(f"  Failed: {e}")
            results[window_size] = None
    if len(results) > 1:
        print("Consistency Check Across Windows")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        if len(valid_results) > 1:
            all_windows = list(valid_results.values())
            consistent_links = all_windows[0].copy()
            for A in all_windows[1:]:
                consistent_links = consistent_links & A
            
            print(f"Links consistent across all windows: {np.sum(consistent_links)}")
            print("Consistent influence matrix:")
            print(consistent_links)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-file", type=str, 
                       default="runs/MADDPG_simple_spread_v3_20251221_155847/trajectory_data.pkl",
                       help="Path to trajectory data file")
    parser.add_argument("--max-lag", type=int, default=5,
                       help="Maximum lag for VAR model")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level")
    parser.add_argument("--check-stationarity", action="store_true", default=True,
                       help="Check stationarity before analysis")
    parser.add_argument("--no-differencing", action="store_true",
                       help="Skip automatic differencing even if non-stationary")
    parser.add_argument("--test-windows", action="store_true",
                       help="Test robustness across different time windows")
    parser.add_argument("--metric", type=str, default="agent_values",
                       choices=["agent_values", "policy_params", "episode_rewards"],
                       help="Which metric to analyze")
    parser.add_argument("--ground-truth", type=str, default=None,
                       help="Path to ground truth influence matrix (.npy or .pkl file)")
    parser.add_argument("--save-results", type=str, default=None,
                       help="Path to save results (influence matrix, metrics, etc.)")
    
    args = parser.parse_args()

    print(f"Loading trajectory data from: {args.trajectory_file}")
    with open(args.trajectory_file, 'rb') as f:
        data = pickle.load(f)

    num_agents = data['num_agents']
    trajectories = []
    
    if args.metric == "agent_values":
        for agent_idx in range(num_agents):
            trajectories.append(data['agent_values'][agent_idx])
    elif args.metric == "policy_params":
        for agent_idx in range(num_agents):
            params = data['policy_params'][agent_idx]
            param_norms = np.linalg.norm(params, axis=1)
            trajectories.append(param_norms)
    elif args.metric == "episode_rewards":
        for agent_idx in range(num_agents):
            trajectories.append(data['episode_rewards'][agent_idx])
    
    trajectories = np.array(trajectories).T
    trajectories = np.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nUsing metric: {args.metric}")
    A_hat = estimate_influence_granger(
        trajectories, 
        max_lag=args.max_lag, 
        alpha=args.alpha,
        check_stationarity=args.check_stationarity,
        make_stationary_if_needed=not args.no_differencing
    )

    if args.test_windows:
        window_results = test_time_windows(
            trajectories, 
            max_lag=args.max_lag, 
            alpha=args.alpha
        )

    A_true = None
    metrics = None

    if args.ground_truth:
        print("Loading Ground Truth")

        try:
            if args.ground_truth.endswith('.npy'):
                A_true = np.load(args.ground_truth)
            elif args.ground_truth.endswith('.pkl'):
                with open(args.ground_truth, 'rb') as f:
                    A_true = pickle.load(f)
            else:
                A_true = np.load(args.ground_truth)
            
            print(f"Ground truth shape: {A_true.shape}")
            print(A_true)

            if A_true.shape != A_hat.shape:
                print(f"Warning: Shape mismatch! A_hat: {A_hat.shape}, A_true: {A_true.shape}")
            else:
                metrics = validate_influence_matrix(A_hat, A_true)
        except Exception as e:
            print(f"Error loading: {e}")
            A_true = None

    if args.save_results:
        results = {
            'influence_matrix': A_hat,
            'num_agents': num_agents,
            'metric': args.metric,
            'max_lag': args.max_lag,
            'alpha': args.alpha,
            'ground_truth': A_true,
            'validation_metrics': metrics
        }
        
        if args.save_results.endswith('.pkl'):
            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f)
        else:
            with open(args.save_results + '.pkl', 'wb') as f:
                pickle.dump(results, f)
        np.save(args.save_results.replace('.pkl', '_A_hat.npy'), A_hat)
        
        print(f"\nResults saved to: {args.save_results}")

