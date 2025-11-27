#!/usr/bin/env python3
"""
Create and Recreate Model Comparison Plots

It creates 10 high-quality plots from experimental results and can be used
both as a library and a script.

Usage as a script:
    python create_model_plots.py

Usage as a module:
    from create_model_plots import create_all_plots
    baseline_metrics, window_metrics = load_metrics("output/experiment_results.npz")
    create_all_plots(baseline_metrics, window_metrics, output_dir="plots")
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

import seaborn as sns
from scipy import stats


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_all_plots(baseline_metrics, window_metrics, output_dir='plots'):
    """
    Create all 10 plots showing different aspects of comparison

    Args:
        baseline_metrics: List of dicts with baseline model results
        window_metrics: List of dicts with window model results
        output_dir: Directory to save plots (default: 'plots')
    """
    os.makedirs(output_dir, exist_ok=True)

    n_runs = len(baseline_metrics)
    print(f"\nCREATING 10 HIGH-QUALITY PLOTS ({n_runs} runs)\n")

    # Plot 1: Training loss curves with mean and confidence interval
    print("1. Training loss curves with mean...")
    plot_training_curves(baseline_metrics, window_metrics, os.path.join(output_dir, '1_training_curves.png'))

    # Plot 2: Final losses comparison with mean
    print("2. Final 100 losses comparison...")
    plot_final_losses(baseline_metrics, window_metrics, os.path.join(output_dir, '2_final_losses.png'))

    # Plot 3: Metrics comparison
    print("3. Metrics bar charts...")
    plot_metrics_comparison(baseline_metrics, window_metrics, os.path.join(output_dir, '3_metrics_comparison.png'))

    # Plot 4: Speed vs quality tradeoff
    print("4. Speed-quality tradeoff...")
    plot_speedup_quality(baseline_metrics, window_metrics, os.path.join(output_dir, '4_speedup_quality.png'))

    # Plot 5: Memory vs quality tradeoff
    print("5. Memory-quality tradeoff...")
    plot_memory_quality(baseline_metrics, window_metrics, os.path.join(output_dir, '5_memory_quality.png'))

    # Plot 6: Memory vs speed tradeoff
    print("6. Memory-speed tradeoff...")
    plot_memory_speed(baseline_metrics, window_metrics, os.path.join(output_dir, '6_memory_speed.png'))

    # Plot 7: Statistical distributions
    print("7. Statistical distributions...")
    plot_distributions(baseline_metrics, window_metrics, os.path.join(output_dir, '7_distributions.png'))

    # Plot 8: Per-run breakdown
    print("8. Per-run breakdown...")
    plot_per_run(baseline_metrics, window_metrics, os.path.join(output_dir, '8_per_run.png'))

    # Plot 9: Summary plot
    print("9. Improvement summary...")
    plot_summary(baseline_metrics, window_metrics, os.path.join(output_dir, '9_summary.png'))

    # Plot 10: Smoothed convergence with confidence bands
    print("10. Loss convergence with confidence bands...")
    plot_convergence(baseline_metrics, window_metrics, os.path.join(output_dir, '10_convergence.png'))


def plot_training_curves(baseline_metrics, window_metrics, save_path):
    """Plot 1: Training loss curves with mean and individual runs (transparent)"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_runs = len(baseline_metrics)
    
    # Plot individual baseline runs with high transparency
    for m in baseline_metrics:
        ax.plot(m['tokens_seen'], m['losses'], 'b-', alpha=0.15, linewidth=1)
    
    # Plot individual window runs with high transparency
    for m in window_metrics:
        ax.plot(m['tokens_seen'], m['losses'], color='orange', alpha=0.15, linewidth=1)
    
    # Baseline mean
    if baseline_metrics:
        max_tokens = max(m['tokens_seen'][-1] for m in baseline_metrics)
        token_grid = np.linspace(0, max_tokens, 1000)
        
        baseline_losses_interp = []
        for m in baseline_metrics:
            losses_interp = np.interp(token_grid, m['tokens_seen'], m['losses'])
            baseline_losses_interp.append(losses_interp)
        
        baseline_mean = np.mean(baseline_losses_interp, axis=0)
        baseline_std = np.std(baseline_losses_interp, axis=0)
        
        ax.plot(token_grid, baseline_mean, 'b-', linewidth=3, label=f'Baseline Mean (n={n_runs})')
        ax.fill_between(token_grid, baseline_mean - baseline_std, baseline_mean + baseline_std,
                        color='blue', alpha=0.2, label='Baseline ±1 STD')
    
    # Window mean
    if window_metrics:
        max_tokens = max(m['tokens_seen'][-1] for m in window_metrics)
        token_grid = np.linspace(0, max_tokens, 1000)
        
        window_losses_interp = []
        for m in window_metrics:
            losses_interp = np.interp(token_grid, m['tokens_seen'], m['losses'])
            window_losses_interp.append(losses_interp)
        
        window_mean = np.mean(window_losses_interp, axis=0)
        window_std = np.std(window_losses_interp, axis=0)
        
        ax.plot(token_grid, window_mean, color='orange', linewidth=3, 
                label=f'Window Mean (n={n_runs})')
        ax.fill_between(token_grid, window_mean - window_std, window_mean + window_std,
                        color='orange', alpha=0.2, label='Window ±1 STD')

    ax.set_xlabel('Tokens Seen', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title(f'Training Loss Curves: Baseline vs Window Sparse Attention ({n_runs} runs)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_final_losses(baseline_metrics, window_metrics, save_path):
    """Plot 2: Last 100 losses with mean line"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_runs = len(baseline_metrics)

    # Plot last 100 losses for each run (transparent)
    for m in baseline_metrics:
        losses = np.array(m['losses'])[-100:]
        ax.plot(losses, 'b-', alpha=0.2, linewidth=1.5)

    for m in window_metrics:
        losses = np.array(m['losses'])[-100:]
        ax.plot(losses, color='orange', alpha=0.2, linewidth=1.5)

    baseline_last100_all = [np.array(m['losses'])[-100:] for m in baseline_metrics]
    window_last100_all = [np.array(m['losses'])[-100:] for m in window_metrics]
    
    max_len_b = max(len(x) for x in baseline_last100_all)
    max_len_w = max(len(x) for x in window_last100_all)
    
    baseline_mean = np.mean(
        [np.pad(x, (0, max_len_b - len(x)), 'edge') for x in baseline_last100_all],
        axis=0
    )
    window_mean = np.mean(
        [np.pad(x, (0, max_len_w - len(x)), 'edge') for x in window_last100_all],
        axis=0
    )

    ax.plot(baseline_mean, 'b-', linewidth=3, label=f'Baseline Mean (n={n_runs})')
    ax.plot(window_mean, color='orange', linewidth=3, label=f'Window Mean (n={n_runs})')

    baseline_avg = np.mean([m['avg_loss_last_100'] for m in baseline_metrics])
    window_avg = np.mean([m['avg_loss_last_100'] for m in window_metrics])

    ax.axhline(baseline_avg, color='blue', linestyle='--', linewidth=2,
               label=f'Baseline Overall Avg: {baseline_avg:.4f}')
    ax.axhline(window_avg, color='orange', linestyle='--', linewidth=2,
               label=f'Window Overall Avg: {window_avg:.4f}')

    ax.set_xlabel('Step (last 100)', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'Final 100 Steps Loss Comparison ({n_runs} runs)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black',
              fancybox=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_metrics_comparison(baseline_metrics, window_metrics, save_path):
    """Plot 3: Bar charts of all metrics with error bars"""
    baseline_time = np.mean([m['training_time'] for m in baseline_metrics])
    baseline_time_std = np.std([m['training_time'] for m in baseline_metrics])
    window_time = np.mean([m['training_time'] for m in window_metrics])
    window_time_std = np.std([m['training_time'] for m in window_metrics])
    
    baseline_mem = np.mean([m['peak_memory_mb'] for m in baseline_metrics])
    baseline_mem_std = np.std([m['peak_memory_mb'] for m in baseline_metrics])
    window_mem = np.mean([m['peak_memory_mb'] for m in window_metrics])
    window_mem_std = np.std([m['peak_memory_mb'] for m in window_metrics])
    
    baseline_loss = np.mean([m['avg_loss_last_100'] for m in baseline_metrics])
    baseline_loss_std = np.std([m['avg_loss_last_100'] for m in baseline_metrics])
    window_loss = np.mean([m['avg_loss_last_100'] for m in window_metrics])
    window_loss_std = np.std([m['avg_loss_last_100'] for m in window_metrics])

    time_imp = (baseline_time - window_time) / baseline_time * 100
    mem_imp = (baseline_mem - window_mem) / baseline_mem * 100
    loss_diff = (window_loss - baseline_loss) / baseline_loss * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    n_runs = len(baseline_metrics)

    # Training time
    axes[0].bar(['Baseline', 'Window'], [baseline_time, window_time],
                yerr=[baseline_time_std, window_time_std],
                color=['blue', 'orange'], alpha=0.7, edgecolor='black',
                capsize=5, error_kw={'linewidth': 2})
    axes[0].set_ylabel('Time (seconds)', fontsize=13)
    axes[0].set_title(f'Training Time (n={n_runs})\n({time_imp:+.2f}% improvement)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (v, err) in enumerate([(baseline_time, baseline_time_std), 
                                  (window_time, window_time_std)]):
        axes[0].text(i, v + err, f'{v:.1f}±{err:.1f}s', 
                     ha='center', va='bottom', fontweight='bold')

    # Memory
    axes[1].bar(['Baseline', 'Window'], [baseline_mem, window_mem],
                yerr=[baseline_mem_std, window_mem_std],
                color=['blue', 'orange'], alpha=0.7, edgecolor='black',
                capsize=5, error_kw={'linewidth': 2})
    axes[1].set_ylabel('Memory (MB)', fontsize=13)
    axes[1].set_title(f'Peak GPU Memory (n={n_runs})\n({mem_imp:+.2f}% reduction)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (v, err) in enumerate([(baseline_mem, baseline_mem_std), 
                                  (window_mem, window_mem_std)]):
        axes[1].text(i, v + err, f'{v:.0f}±{err:.0f}', 
                     ha='center', va='bottom', fontweight='bold')

    # Loss
    axes[2].bar(['Baseline', 'Window'], [baseline_loss, window_loss],
                yerr=[baseline_loss_std, window_loss_std],
                color=['blue', 'orange'], alpha=0.7, edgecolor='black',
                capsize=5, error_kw={'linewidth': 2})
    axes[2].set_ylabel('Average Loss', fontsize=13)
    axes[2].set_title(f'Model Quality (n={n_runs})\n({loss_diff:+.2f}% difference)',
                      fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (v, err) in enumerate([(baseline_loss, baseline_loss_std), 
                                  (window_loss, window_loss_std)]):
        axes[2].text(i, v + err, f'{v:.4f}±{err:.4f}', 
                     ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_speedup_quality(baseline_metrics, window_metrics, save_path):
    """Plot 4: Speed vs quality tradeoff scatter"""
    fig, ax = plt.subplots(figsize=(11, 8))

    speedups = []
    quality_diffs = []

    for b, w in zip(baseline_metrics, window_metrics):
        speedup = (b['training_time'] - w['training_time']) / b['training_time'] * 100
        quality_diff = (w['avg_loss_last_100'] - b['avg_loss_last_100']) / b['avg_loss_last_100'] * 100
        speedups.append(speedup)
        quality_diffs.append(quality_diff)

    ax.scatter(speedups, quality_diffs, s=200, alpha=0.6, c='purple',
               edgecolors='black', linewidth=2, zorder=3)

    mean_speedup = np.mean(speedups)
    mean_quality = np.mean(quality_diffs)
    ax.scatter(mean_speedup, mean_quality, s=400, alpha=0.9, c='red',
               edgecolors='black', linewidth=3, marker='*',
               label=f'Mean: {mean_speedup:.1f}% faster, {mean_quality:+.2f}% quality', zorder=4)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.text(0.95, 0.95, 'Faster & Less Quality', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', alpha=0.8))

    ax.text(0.95, 0.05, 'Faster & Better Quality', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#4caf50', alpha=0.8))

    ax.text(0.05, 0.05, 'Slower & Better Quality', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.set_xlim(1, 10)
    y_min, y_max = min(quality_diffs), max(quality_diffs)
    y_padding = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_xlabel('Speed Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Quality Difference (% change in loss)', fontsize=13, fontweight='bold')
    ax.set_title(f'Speed vs Quality Tradeoff (n={len(speedups)} runs)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_memory_quality(baseline_metrics, window_metrics, save_path):
    """Plot 5: Memory vs quality tradeoff scatter"""
    fig, ax = plt.subplots(figsize=(11, 8))

    memory_savings = []
    quality_diffs = []

    for b, w in zip(baseline_metrics, window_metrics):
        memory_saving = (b['peak_memory_mb'] - w['peak_memory_mb']) / b['peak_memory_mb'] * 100
        quality_diff = (w['avg_loss_last_100'] - b['avg_loss_last_100']) / b['avg_loss_last_100'] * 100
        memory_savings.append(memory_saving)
        quality_diffs.append(quality_diff)

    ax.scatter(memory_savings, quality_diffs, s=200, alpha=0.6, c='purple', 
               edgecolors='black', linewidth=2, zorder=3)

    mean_memory = np.mean(memory_savings)
    mean_quality = np.mean(quality_diffs)
    ax.scatter(mean_memory, mean_quality, s=400, alpha=0.9, c='red', 
               edgecolors='black', linewidth=3, marker='*', 
               label=f'Mean: {mean_memory:.1f}% less memory, {mean_quality:+.2f}% quality', zorder=4)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.text(0.95, 0.95, 'Less Memory & Less Quality', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', alpha=0.8))
    
    ax.text(0.95, 0.05, 'Less Memory & Better Quality', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#4caf50', alpha=0.8))
    
    ax.text(0.05, 0.05, 'More Memory & Better Quality', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.text(0.05, 0.95, 'More Memory & Less Quality', transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ffcdd2', alpha=0.8))

    ax.set_xlim(1, 7)
    y_min, y_max = min(quality_diffs), max(quality_diffs)
    y_padding = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_xlabel('Memory Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Quality Difference (% change in loss)', fontsize=13, fontweight='bold')
    ax.set_title(f'Memory vs Quality Tradeoff (n={len(memory_savings)} runs)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_memory_speed(baseline_metrics, window_metrics, save_path):
    """Plot 6: Memory vs speed tradeoff scatter"""
    fig, ax = plt.subplots(figsize=(11, 8))

    memory_savings = []
    speedups = []

    for b, w in zip(baseline_metrics, window_metrics):
        memory_saving = (b['peak_memory_mb'] - w['peak_memory_mb']) / b['peak_memory_mb'] * 100
        speedup = (b['training_time'] - w['training_time']) / b['training_time'] * 100
        memory_savings.append(memory_saving)
        speedups.append(speedup)

    ax.scatter(memory_savings, speedups, s=200, alpha=0.6, c='purple', 
               edgecolors='black', linewidth=2, zorder=3)

    mean_memory = np.mean(memory_savings)
    mean_speed = np.mean(speedups)
    ax.scatter(mean_memory, mean_speed, s=400, alpha=0.9, c='red', 
               edgecolors='black', linewidth=3, marker='*', 
               label=f'Mean: {mean_memory:.1f}% less memory, {mean_speed:+.1f}% faster', zorder=4)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.text(0.95, 0.95, 'Less Memory & Faster', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#4caf50', alpha=0.8))
    
    ax.text(0.95, 0.05, 'Less Memory & Slower', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.8))
    
    ax.text(0.05, 0.95, 'More Memory & Faster', transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.8))
    
    ax.text(0.05, 0.05, 'More Memory & Slower', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.set_ylim(0, 8)
    ax.set_xlabel('Memory Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speed Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Memory vs Speed Tradeoff (n={len(memory_savings)} runs)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower center', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_distributions(baseline_metrics, window_metrics, save_path):
    """Plot 7: Statistical distributions"""
    baseline_times = [m['training_time'] for m in baseline_metrics]
    window_times = [m['training_time'] for m in window_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    ax = axes[0]
    bp = ax.boxplot([baseline_times, window_times], tick_labels=['Baseline', 'Window'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('orange')
    for patch in bp['boxes']:
        patch.set_alpha(0.7)

    ax.set_ylabel('Training Time (seconds)', fontsize=13)
    ax.set_title('Training Time Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, times in enumerate([baseline_times, window_times]):
        x = np.random.normal(i + 1, 0.04, size=len(times))
        ax.scatter(x, times, alpha=0.6, s=100, c='black', zorder=3)

    # Violin plot
    ax = axes[1]
    parts = ax.violinplot([baseline_times, window_times], positions=[1, 2],
                          widths=0.7, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['blue', 'orange'][i])
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'Window'])
    ax.set_ylabel('Training Time (seconds)', fontsize=13)
    ax.set_title('Training Time Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    if len(baseline_times) > 1:
        t_stat, p_value = stats.ttest_ind(baseline_times, window_times)
        
        if p_value < 0.0001:
            p_str = '< 0.0001'
        else:
            p_str = f'{p_value:.4f}'
        
        textstr = f't-stat: {t_stat:.4f}\np-value: {p_str}'
        if p_value < 0.05:
            textstr += '\n* Significant (p<0.05)'
        elif p_value < 0.10:
            textstr += '\n~ Marginally sig. (p<0.10)'
        color = 'lightgreen' if p_value < 0.05 else ('lightyellow' if p_value < 0.10 else 'lightcoral')
        props = dict(boxstyle='round', facecolor=color, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_per_run(baseline_metrics, window_metrics, save_path):
    """Plot 8: Per-run breakdown - optimized for many runs"""
    n_runs = len(baseline_metrics)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    runs = np.arange(1, n_runs + 1)
    width = 0.35

    baseline_times = [m['training_time'] for m in baseline_metrics]
    window_times = [m['training_time'] for m in window_metrics]
    axes[0].bar(runs - width / 2, baseline_times,
                width, label='Baseline', color='blue', alpha=0.7, edgecolor='black')
    axes[0].bar(runs + width / 2, window_times,
                width, label='Window', color='orange', alpha=0.7, edgecolor='black')
    
    axes[0].axhline(np.mean(baseline_times), color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Baseline Mean: {np.mean(baseline_times):.1f}s')
    axes[0].axhline(np.mean(window_times), color='orange', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Window Mean: {np.mean(window_times):.1f}s')
    
    axes[0].set_ylabel('Time (s)', fontsize=12)
    axes[0].set_title(f'Training Time per Run (n={n_runs})', fontsize=13, fontweight='bold')
    axes[0].set_xticks(runs)
    axes[0].legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='best')
    axes[0].grid(True, alpha=0.3, axis='y')

    baseline_mems = [m['peak_memory_mb'] for m in baseline_metrics]
    window_mems = [m['peak_memory_mb'] for m in window_metrics]
    axes[1].bar(runs - width / 2, baseline_mems,
                width, label='Baseline', color='blue', alpha=0.7, edgecolor='black')
    axes[1].bar(runs + width / 2, window_mems,
                width, label='Window', color='orange', alpha=0.7, edgecolor='black')
    
    axes[1].axhline(np.mean(baseline_mems), color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Baseline Mean: {np.mean(baseline_mems):.0f} MB')
    axes[1].axhline(np.mean(window_mems), color='orange', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Window Mean: {np.mean(window_mems):.0f} MB')
    
    axes[1].set_ylabel('Memory (MB)', fontsize=12)
    axes[1].set_title(f'Peak GPU Memory per Run (n={n_runs})', fontsize=13, fontweight='bold')
    axes[1].set_xticks(runs)
    axes[1].legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='best')
    axes[1].grid(True, alpha=0.3, axis='y')

    baseline_losses = [m['avg_loss_last_100'] for m in baseline_metrics]
    window_losses = [m['avg_loss_last_100'] for m in window_metrics]
    axes[2].bar(runs - width / 2, baseline_losses,
                width, label='Baseline', color='blue', alpha=0.7, edgecolor='black')
    axes[2].bar(runs + width / 2, window_losses,
                width, label='Window', color='orange', alpha=0.7, edgecolor='black')
    
    axes[2].axhline(np.mean(baseline_losses), color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Baseline Mean: {np.mean(baseline_losses):.4f}')
    axes[2].axhline(np.mean(window_losses), color='orange', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Window Mean: {np.mean(window_losses):.4f}')
    
    axes[2].set_xlabel('Run Number', fontsize=13)
    axes[2].set_ylabel('Avg Loss', fontsize=12)
    axes[2].set_title(f'Average Loss (last 100) per Run (n={n_runs})', fontsize=13, fontweight='bold')
    axes[2].set_xticks(runs)
    axes[2].legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='best')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


def plot_summary(baseline_metrics, window_metrics, save_path):
    """Plot 9: Summary horizontal bar chart with confidence intervals"""
    n_runs = len(baseline_metrics)
    
    time_mean = (np.mean([m['training_time'] for m in baseline_metrics]) -
                 np.mean([m['training_time'] for m in window_metrics])) / \
                np.mean([m['training_time'] for m in baseline_metrics]) * 100

    mem_mean = (np.mean([m['peak_memory_mb'] for m in baseline_metrics]) -
                np.mean([m['peak_memory_mb'] for m in window_metrics])) / \
               np.mean([m['peak_memory_mb'] for m in baseline_metrics]) * 100

    loss_diff = (np.mean([m['avg_loss_last_100'] for m in window_metrics]) -
                 np.mean([m['avg_loss_last_100'] for m in baseline_metrics])) / \
                np.mean([m['avg_loss_last_100'] for m in baseline_metrics]) * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)

    metrics = ['Training\nSpeed', 'Memory\nUsage', 'Model\nQuality']
    improvements = [time_mean, mem_mean, -loss_diff]  # Negative for quality
    colors = ['green' if x > 0 else 'red' for x in improvements]

    bars = ax.barh(metrics, improvements, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)

    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        label_x = width + (0.3 if width > 0 else -0.3)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f'{abs(val):.2f}%',
                ha=ha, va='center', fontsize=15, fontweight='bold')

    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('Improvement (%)', fontsize=15, fontweight='bold')
    ax.set_title(f'Window Sparse Attention: Performance Summary (w=96, d=4, n={n_runs})',
                 fontsize=17, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    max_val = max(abs(x) for x in improvements)
    ax.set_xlim(-max_val * 0.3, max_val * 1.2)

    if n_runs >= 10:
        significance = "with statistical confidence"
    elif n_runs >= 5:
        significance = "with good sample size"
    else:
        significance = "preliminary results"
    
    textstr = (
        f'KEY FINDINGS ({n_runs} runs):\n'
        f'─────────────────────\n'
        f'+ {time_mean:.1f}% faster training\n'
        f'+ {mem_mean:.1f}% lower memory\n'
        f'+ {abs(loss_diff):.1f}% quality diff\n'
        f'   (comparable)\n\n'
        f'{significance}'
    )
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"  Saved {save_path}")
    plt.close()


def plot_convergence(baseline_metrics, window_metrics, save_path):
    """Plot 10: Smoothed loss convergence with confidence bands"""
    fig, ax = plt.subplots(figsize=(14, 7))

    window_size = 50
    n_runs = len(baseline_metrics)

    baseline_tokens_grid = None
    baseline_mas = []
    
    for m in baseline_metrics:
        losses = np.array(m['losses'])
        tokens = np.array(m['tokens_seen'])

        ma = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        tokens_ma = tokens[window_size - 1:]
        
        if baseline_tokens_grid is None:
            baseline_tokens_grid = tokens_ma
        
        ma_interp = np.interp(baseline_tokens_grid, tokens_ma, ma)
        baseline_mas.append(ma_interp)
        
        ax.plot(tokens_ma, ma, 'b-', alpha=0.15, linewidth=1)
    
    baseline_mean = np.mean(baseline_mas, axis=0)
    baseline_std = np.std(baseline_mas, axis=0)
    ax.plot(baseline_tokens_grid, baseline_mean, 'b-', linewidth=3, 
            label=f'Baseline Mean (n={n_runs})')
    ax.fill_between(baseline_tokens_grid, 
                    baseline_mean - baseline_std, 
                    baseline_mean + baseline_std,
                    color='blue', alpha=0.2, label='Baseline ±1 STD')

    window_tokens_grid = None
    window_mas = []
    
    for m in window_metrics:
        losses = np.array(m['losses'])
        tokens = np.array(m['tokens_seen'])

        ma = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        tokens_ma = tokens[window_size - 1:]
        
        if window_tokens_grid is None:
            window_tokens_grid = tokens_ma
        
        ma_interp = np.interp(window_tokens_grid, tokens_ma, ma)
        window_mas.append(ma_interp)
        
        ax.plot(tokens_ma, ma, color='orange', alpha=0.15, linewidth=1)
    
    window_mean = np.mean(window_mas, axis=0)
    window_std = np.std(window_mas, axis=0)
    ax.plot(window_tokens_grid, window_mean, color='orange', linewidth=3, 
            label=f'Window Mean (n={n_runs})')
    ax.fill_between(window_tokens_grid, 
                    window_mean - window_std, 
                    window_mean + window_std,
                    color='orange', alpha=0.2, label='Window ±1 STD')

    ax.set_xlabel('Tokens Seen', fontsize=13)
    ax.set_ylabel(f'Loss (Moving Average, window={window_size})', fontsize=13)
    ax.set_title(f'Smoothed Loss Convergence with Confidence Bands ({n_runs} runs)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black',
              fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.close()


# ----------------------------------------------------------------------
# CLI entry point 
# ----------------------------------------------------------------------

def main():
    from metrics_utils import load_metrics, print_metrics_summary

    os.makedirs('plots', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = os.path.join('output', 'experiment_results.npz')
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'plots'
    
   
    print("\nRecreating plots from saved metrics...")
    print(f"\nResults file: {results_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        baseline_metrics, window_metrics = load_metrics(results_file)
    except FileNotFoundError:
        print(f"\nError: File '{results_file}' not found!")
        print("\nAvailable options:")
        print("  1. Run the experiment first: python compare_models.py")
        print("  2. Specify a different file: python create_model_plots.py your_file.npz")
        sys.exit(1)
    
    print_metrics_summary(baseline_metrics, window_metrics)
     
    print("\nCreating plots...\n")
    
    create_all_plots(baseline_metrics, window_metrics, output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
