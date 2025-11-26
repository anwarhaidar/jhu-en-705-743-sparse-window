#!/usr/bin/env python3
"""
Recreate plots from saved metrics

This script loads previously saved experiment results and recreates all plots
without needing to retrain the models.

Usage:
    python recreate_plots.py [results_file.npz] [output_dir]
    
Examples:
    # Use default files
    python recreate_plots.py
    
    # Specify results file
    python recreate_plots.py my_experiment.npz
    
    # Specify results file and output directory
    python recreate_plots.py my_experiment.npz my_plots/
"""

import sys
import os
from metrics_utils import load_metrics, print_metrics_summary
from create_plots_improved import create_all_plots


def main():
    # Parse command line arguments
    # Create plots directory if it doesn't exist
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
    
    print("="*80)
    print("RECREATING PLOTS FROM SAVED METRICS")
    print("="*80)
    print(f"\nResults file: {results_file}")
    print(f"Output directory: {output_dir}")
    
    # Load metrics
    try:
        baseline_metrics, window_metrics = load_metrics(results_file)
    except FileNotFoundError:
        print(f"\n❌ Error: File '{results_file}' not found!")
        print("\nAvailable options:")
        print("  1. Run the experiment first: python compare_models_10runs.py")
        print("  2. Specify a different file: python recreate_plots.py your_file.npz")
        sys.exit(1)
    
    # Print summary
    print_metrics_summary(baseline_metrics, window_metrics)
    
    # Create plots
    print(f"\n{'='*80}")
    print("CREATING PLOTS")
    print(f"{'='*80}\n")
    
    create_all_plots(baseline_metrics, window_metrics, output_dir)
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    print(f"\nAll plots saved to '{output_dir}/'")
    print(f"Metrics loaded from '{results_file}'")
    print("\nYou can modify plot settings in 'create_plots_improved.py' and")
    print("rerun this script to update the plots without retraining!")


if __name__ == "__main__":
    main()
