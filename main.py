#!/usr/bin/env python3
"""
Sliding-Window Sparse Attention in GPT Models
Main Experiment Script

This script runs the complete experimental comparison between baseline GPT
and sparse attention GPT models. It performs 10 randomized runs with full
statistical analysis and visualization.

Author: Anwar Sleiman Haidar
Course: EN.705.743 - ChatGPT from Scratch
Institution: Johns Hopkins University
Date: October 13, 2025

Usage:
    python main.py

Expected runtime: ~30 minutes per run, ~5 hours total for 10 runs (GPU required)
"""

import torch
import os
import sys

# Import the comparison framework
from compare_models_10runs import compare_models_improved
from create_plots_improved import create_all_plots

def check_dependencies():
    """
    Check if all required dependencies are available.
    """
    print("=" * 80)
    print("DEPENDENCY CHECK")
    print("=" * 80)
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not found. Install: pip install matplotlib")
        sys.exit(1)
    
    try:
        import seaborn
        print("✓ seaborn available")
    except ImportError:
        print("✗ seaborn not found. Install: pip install seaborn")
        sys.exit(1)
    
    try:
        import scipy
        print("✓ scipy available")
    except ImportError:
        print("✗ scipy not found. Install: pip install scipy")
        sys.exit(1)
    
    try:
        from tqdm import tqdm
        print("✓ tqdm available")
    except ImportError:
        print("✗ tqdm not found. Install: pip install tqdm")
        sys.exit(1)
    
    # Check for CUDA - REQUIRED
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n❌ ERROR: CUDA not available!")
        print("This experiment requires a GPU. CPU training would take 50+ hours.")
        print("\nPlease:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA toolkit")
        print("  3. Install PyTorch with CUDA support")
        print("  4. Verify with: python -c 'import torch; print(torch.cuda.is_available())'")
        sys.exit(1)
    
    print()


def check_data_files():
    """
    Check if required data files are present.
    """
    print("=" * 80)
    print("DATA FILE CHECK")
    print("=" * 80)
    
    required_files = [
        'training_data.npy',
        'gpt.py',
        'gpt_win.py',
        'embedding.py',
        'linear.py',
        'train_model.py',
        'compare_models_10runs.py',
        'create_plots_improved.py',
        'metrics_utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Error: Missing required files: {missing_files}")
        print("Please ensure all files are in the current directory.")
        sys.exit(1)
    
    print()


def print_experiment_info():
    """
    Print information about the experiment setup.
    """
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print("""
This experiment will:

1. Run 10 randomized comparisons between baseline and sparse attention GPT
2. Train each model on Shakespeare text data
3. Track training time, memory usage, and model quality
4. Perform statistical analysis (t-tests, Cohen's d)
5. Generate 8 publication-quality plots
6. Save all metrics for later analysis

Model Configuration:
  - Architecture: GPT with 8 layers, 16 heads, d_model=512
  - Training: 8 batch size, 3e-4 learning rate, 500 warmup steps
  - Sparse attention: window_size=96, dilation=4
  
Expected Results:
  - Training speedup: 5-10%
  - Memory reduction: 4-5%
  - Quality difference: <1% (negligible)
  
Expected Runtime:
  - Per single run (both models): ~30 minutes
  - Total for 10 runs: ~5 hours
  - GPU Required: NVIDIA GPU with 8GB+ VRAM
  - CPU: Not supported (would take 50+ hours)

Statistical Validation:
  - 10 runs provide high confidence (n=10)
  - Randomized order prevents bias
  - t-tests for significance (α=0.05)
  - Cohen's d for effect size
    """)
    print("=" * 80)
    print()


def main():
    """
    Main entry point for the experiment.
    """
    print("\n" + "=" * 80)
    print("SLIDING-WINDOW SPARSE ATTENTION IN GPT MODELS")
    print("Practical Implementation and Empirical Evaluation")
    print("=" * 80)
    print()
    print("Author: Anwar Sleiman Haidar")
    print("Course: EN.705.743 - ChatGPT from Scratch")
    print("Institution: Johns Hopkins University")
    print("Date: November 18, 2025")
    print()
    
    # Check dependencies
    check_dependencies()
    
    # Check data files
    check_data_files()
    
    # Print experiment info
    print_experiment_info()
    
    # Confirm before starting
    print("WARNING: This experiment takes approximately 5 HOURS to complete!")
    print("Each run takes ~30 minutes, and we perform 10 runs for statistical validity.\n")
    response = input("Ready to start experiment? This will take ~5 hours. [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("\nExperiment cancelled.")
        print("Tip: You can run with fewer iterations by editing num_runs=10 in main.py (line 200)")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT")
    print("=" * 80)
    print()
    
    try:
        # Run the main comparison experiment
        # This performs 10 runs with randomized order
        baseline_metrics, window_metrics = compare_models_improved(
            num_runs=10,          # Number of iterations
            randomize_order=True  # Randomize baseline/window order
        )

        create_all_plots(baseline_metrics, window_metrics)

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Generated outputs:")
        print("  ✓ output/experiment_results.npz - Saved metrics")
        print("  ✓ output/experiment_results.json - Human-readable metrics")
        print("  ✓ plots/ - 8 visualization plots")
        print("  ✓ output/improved_comparison_results.txt - Text summary")
        print()
        print("Next steps:")
        print("  1. Check plots/ directory for visualizations")
        print("  2. Review markdown/improved_comparison_results.txt for summary")
        print("  3. Use recreate_plots.py to regenerate plots with different styling")
        print()
        print("To recreate plots anytime:")
        print("  python recreate_plots.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Experiment interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
