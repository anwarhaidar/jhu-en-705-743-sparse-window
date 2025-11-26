"""
Utilities for saving and loading model comparison metrics

This module provides functions to:
1. Save metrics from experiments to disk
2. Load metrics from disk
3. Recreate plots without retraining

Usage:
    # After running comparison
    save_metrics(baseline_metrics, window_metrics, 'results.npz')
    
    # Later, to recreate plots
    baseline_metrics, window_metrics = load_metrics('results.npz')
    create_all_plots(baseline_metrics, window_metrics)
"""

import numpy as np
import json

from create_plots_improved import create_all_plots

def save_metrics(baseline_metrics, window_metrics, filepath='experiment_results.npz'):
    """
    Save model comparison metrics to disk
    
    Args:
        baseline_metrics: List of dicts with baseline model results
        window_metrics: List of dicts with window model results
        filepath: Path to save the data (default: 'experiment_results.npz')
    
    Example:
        >>> save_metrics(baseline_metrics, window_metrics, 'my_results.npz')
        Saved metrics to my_results.npz (10 baseline runs, 10 window runs)
    """
    # Convert list of dicts to dict of lists for easier numpy storage
    baseline_data = {
        'losses': [m['losses'] for m in baseline_metrics],
        'tokens_seen': [m['tokens_seen'] for m in baseline_metrics],
        'training_time': [m['training_time'] for m in baseline_metrics],
        'peak_memory_mb': [m['peak_memory_mb'] for m in baseline_metrics],
        'final_loss': [m['final_loss'] for m in baseline_metrics],
        'avg_loss_last_100': [m['avg_loss_last_100'] for m in baseline_metrics],
        'total_steps': [m['total_steps'] for m in baseline_metrics],
        'total_tokens': [m['total_tokens'] for m in baseline_metrics],
    }
    
    window_data = {
        'losses': [m['losses'] for m in window_metrics],
        'tokens_seen': [m['tokens_seen'] for m in window_metrics],
        'training_time': [m['training_time'] for m in window_metrics],
        'peak_memory_mb': [m['peak_memory_mb'] for m in window_metrics],
        'final_loss': [m['final_loss'] for m in window_metrics],
        'avg_loss_last_100': [m['avg_loss_last_100'] for m in window_metrics],
        'total_steps': [m['total_steps'] for m in window_metrics],
        'total_tokens': [m['total_tokens'] for m in window_metrics],
    }
    
    # Save using numpy's compressed format
    np.savez_compressed(
        filepath,
        baseline_losses=baseline_data['losses'],
        baseline_tokens_seen=baseline_data['tokens_seen'],
        baseline_training_time=baseline_data['training_time'],
        baseline_peak_memory_mb=baseline_data['peak_memory_mb'],
        baseline_final_loss=baseline_data['final_loss'],
        baseline_avg_loss_last_100=baseline_data['avg_loss_last_100'],
        baseline_total_steps=baseline_data['total_steps'],
        baseline_total_tokens=baseline_data['total_tokens'],
        window_losses=window_data['losses'],
        window_tokens_seen=window_data['tokens_seen'],
        window_training_time=window_data['training_time'],
        window_peak_memory_mb=window_data['peak_memory_mb'],
        window_final_loss=window_data['final_loss'],
        window_avg_loss_last_100=window_data['avg_loss_last_100'],
        window_total_steps=window_data['total_steps'],
        window_total_tokens=window_data['total_tokens'],
    )
    
    n_baseline = len(baseline_metrics)
    n_window = len(window_metrics)
    print(f"\n✓ Saved metrics to {filepath}")
    print(f"  - {n_baseline} baseline runs")
    print(f"  - {n_window} window runs")
    
    # Also save as JSON for human readability
    json_path = str(filepath).replace('.npz', '.json')
    json_data = {
        'baseline': baseline_data,
        'window': window_data,
        'metadata': {
            'n_baseline_runs': n_baseline,
            'n_window_runs': n_window,
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_lists(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_data = convert_to_lists(json_data)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  - Human-readable version: {json_path}")


def load_metrics(filepath='experiment_results.npz'):
    """
    Load model comparison metrics from disk
    
    Args:
        filepath: Path to the saved data (default: 'experiment_results.npz')
    
    Returns:
        baseline_metrics: List of dicts with baseline model results
        window_metrics: List of dicts with window model results
    
    Example:
        >>> baseline_metrics, window_metrics = load_metrics('my_results.npz')
        Loaded metrics from my_results.npz (10 baseline runs, 10 window runs)
        >>> create_all_plots(baseline_metrics, window_metrics)
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Reconstruct baseline metrics
    n_baseline = len(data['baseline_training_time'])
    baseline_metrics = []
    for i in range(n_baseline):
        metrics = {
            'losses': list(data['baseline_losses'][i]),
            'tokens_seen': list(data['baseline_tokens_seen'][i]),
            'training_time': float(data['baseline_training_time'][i]),
            'peak_memory_mb': float(data['baseline_peak_memory_mb'][i]),
            'final_loss': float(data['baseline_final_loss'][i]),
            'avg_loss_last_100': float(data['baseline_avg_loss_last_100'][i]),
            'total_steps': int(data['baseline_total_steps'][i]),
            'total_tokens': int(data['baseline_total_tokens'][i]),
        }
        baseline_metrics.append(metrics)
    
    # Reconstruct window metrics
    n_window = len(data['window_training_time'])
    window_metrics = []
    for i in range(n_window):
        metrics = {
            'losses': list(data['window_losses'][i]),
            'tokens_seen': list(data['window_tokens_seen'][i]),
            'training_time': float(data['window_training_time'][i]),
            'peak_memory_mb': float(data['window_peak_memory_mb'][i]),
            'final_loss': float(data['window_final_loss'][i]),
            'avg_loss_last_100': float(data['window_avg_loss_last_100'][i]),
            'total_steps': int(data['window_total_steps'][i]),
            'total_tokens': int(data['window_total_tokens'][i]),
        }
        window_metrics.append(metrics)
    
    print(f"\n✓ Loaded metrics from {filepath}")
    print(f"  - {n_baseline} baseline runs")
    print(f"  - {n_window} window runs")
    
    return baseline_metrics, window_metrics


def print_metrics_summary(baseline_metrics, window_metrics):
    """
    Print a summary of loaded metrics
    
    Args:
        baseline_metrics: List of dicts with baseline model results
        window_metrics: List of dicts with window model results
    """
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    n_runs = len(baseline_metrics)
    
    baseline_times = [m['training_time'] for m in baseline_metrics]
    window_times = [m['training_time'] for m in window_metrics]
    
    baseline_memories = [m['peak_memory_mb'] for m in baseline_metrics]
    window_memories = [m['peak_memory_mb'] for m in window_metrics]
    
    baseline_losses = [m['avg_loss_last_100'] for m in baseline_metrics]
    window_losses = [m['avg_loss_last_100'] for m in window_metrics]
    
    print(f"\nNumber of runs: {n_runs}")
    print(f"\nTraining Time:")
    print(f"  Baseline: {np.mean(baseline_times):.2f} ± {np.std(baseline_times):.2f}s")
    print(f"  Window:   {np.mean(window_times):.2f} ± {np.std(window_times):.2f}s")
    
    time_imp = (np.mean(baseline_times) - np.mean(window_times)) / np.mean(baseline_times) * 100
    print(f"  Improvement: {time_imp:+.2f}%")
    
    print(f"\nPeak Memory:")
    print(f"  Baseline: {np.mean(baseline_memories):.2f} ± {np.std(baseline_memories):.2f} MB")
    print(f"  Window:   {np.mean(window_memories):.2f} ± {np.std(window_memories):.2f} MB")
    
    mem_imp = (np.mean(baseline_memories) - np.mean(window_memories)) / np.mean(baseline_memories) * 100
    print(f"  Improvement: {mem_imp:+.2f}%")
    
    print(f"\nAverage Loss (last 100):")
    print(f"  Baseline: {np.mean(baseline_losses):.4f} ± {np.std(baseline_losses):.4f}")
    print(f"  Window:   {np.mean(window_losses):.4f} ± {np.std(window_losses):.4f}")
    
    loss_diff = (np.mean(window_losses) - np.mean(baseline_losses)) / np.mean(baseline_losses) * 100
    print(f"  Difference: {loss_diff:+.2f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print(__doc__)
    print("\nExample usage:")
    print("  # Save metrics after experiment")
    print("  save_metrics(baseline_metrics, window_metrics, 'my_experiment.npz')")
    print()
    print("  # Load metrics later")
    print("  baseline_metrics, window_metrics = load_metrics('my_experiment.npz')")
    print()
    print("  # Print summary")
    print("  print_metrics_summary(baseline_metrics, window_metrics)")
    print()
    print("  # Recreate plots")
    print("  from create_plots_improved import create_all_plots")
    print("  create_all_plots(baseline_metrics, window_metrics)")
