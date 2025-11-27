# Sliding-Window Sparse Attention in GPT Models

**Author:** Anwar Sleiman Haidar  
**Course:** EN.705.743 - ChatGPT from Scratch: Building and Training Large Language Models  
**Institution:** Johns Hopkins University  
**Date:** November 2025

## Overview

This project implements and evaluates a GPT variant with sliding-window sparse attention as a drop-in replacement for standard full self-attention. The implementation achieves measurable efficiency gains (6.27% training speedup, 4.55% memory reduction) with negligible quality loss (0.48%), all within pure PyTorch without custom CUDA kernels.

## Key Results

**From 10 randomized runs with statistical validation:**

- **Training Speed:** 6.27% faster (p < 0.0001, statistically significant)
- **Memory Usage:** 4.55% reduction in peak GPU memory
- **Model Quality:** 0.48% loss difference (negligible, comparable quality)

## Project Structure

```
chatGPT/
├── main.py                        # Entry point - wrapper around compare_models.py
├── gpt_win.py                     # Sparse attention GPT (inherits from gpt.py)
├── compare_models.py              # Core experiment logic (10-run comparison)
├── create_model_plots.py          # Visualization system (10 plots)
├── analyze_hyperparameters.py     # Hyperparameter configuration analyzer
├── metrics_utils.py               # Save/load experiment data
│
├── gpt.py                         # Baseline GPT implementation (course code)
├── embedding.py                   # Custom embedding layer (course code)
├── linear.py                      # Custom linear layer (course code)
├── train_model.py                 # Training utilities (course code)
│
└── training_data.npy              # Required (generate using course code if missing)
```

### Sample Outputs (Generated - Can Be Safely Deleted)

Running the experiment generates two directories:

**`plots/`** - 12 visualization files:
- `1_training_curves.png` - Loss vs tokens with confidence bands
- `2_final_losses.png` - Last 100 steps comparison
- `3_metrics_comparison.png` - Bar charts with error bars
- `4_speedup_quality.png` - Speed vs quality tradeoff scatter
- `5_memory_quality.png` - Memory vs quality tradeoff scatter
- `6_memory_speed.png` - Memory vs speed tradeoff scatter
- `7_distributions.png` - Statistical distributions (box/violin)
- `8_per_run.png` - Per-run breakdown with means
- `9_summary.png` - Overall improvement summary
- `10_convergence.png` - Smoothed loss with confidence bands
- `hyperparameter_comparison.png` - Attention patterns (from analyzer)
- `sparsity_speedup_tradeoff.png` - Sparsity vs speedup (from analyzer)

**`output/`** - 5 data/results files:
- `experiment_results.npz` - Saved metrics from runs (NumPy format)
- `experiment_results.json` - Human-readable metrics
- `window_model_weights.pt` - Trained sparse model weights
- `improved_comparison_results.txt` - Text summary of results
- `hyperparameter_analysis.txt` - Hyperparameter analysis report

## File Descriptions

### Project-Specific Files (New Code)

| File | Description |
|------|-------------|
| `main.py` | Entry point wrapper that invokes `compare_models.py` |
| `gpt_win.py` | Sparse attention implementation using **inheritance** from `gpt.py`. Contains `GPTWindowModel` (extends `GPTModel`), `TransformerDecoderBlockWindow` (extends `TransformerDecoderBlock`), and `CustomWindowMHA` for sliding-window + dilated attention |
| `compare_models.py` | Core experiment script that trains both models 10 times with randomized order, GPU cleanup, and statistical analysis. Internally calls `create_model_plots.py` |
| `create_model_plots.py` | Generates 10 publication-quality plots. Can be run independently to recreate plots from saved metrics |
| `analyze_hyperparameters.py` | Independent tool for exploring window_size and dilation configurations. Generates attention pattern visualizations and sparsity/speedup estimates |
| `metrics_utils.py` | Utilities for saving/loading experiment metrics |

### Course Files (Unmodified)

| File | Description |
|------|-------------|
| `gpt.py` | Baseline GPT with full self-attention (`GPTModel`, `TransformerDecoderBlock`, `CustomMHA`) |
| `embedding.py` | Custom embedding layer implementation |
| `linear.py` | Custom linear transformation layer |
| `train_model.py` | Training utilities and cosine learning rate scheduler |

### Data Requirements

| File | Description |
|------|-------------|
| `training_data.npy` | **Required.** Tokenized training dataset. Generate using course implementation code if not present |

## Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib seaborn scipy tqdm --break-system-packages
```

Ensure `training_data.npy` exists in the project directory.

### Run Main Experiment

```bash
python main.py
```

This executes `compare_models.py` which:
1. Trains baseline and sparse models 10 times each (20 total training runs)
2. Randomizes training order to avoid systematic bias
3. Performs GPU cleanup between runs
4. Saves metrics to `output/experiment_results.npz`
5. Generates all 10 plots in `plots/`
6. Outputs statistical analysis to console and `output/improved_comparison_results.txt`

### Recreate Plots (Without Retraining)

```bash
python create_model_plots.py
```

Or with custom paths:

```bash
python create_model_plots.py output/experiment_results.npz plots/
```

### Analyze Hyperparameters

```bash
# Compare 8 different configurations
python analyze_hyperparameters.py

# Test a specific configuration
python analyze_hyperparameters.py 96 4        # window=96, dilation=4
python analyze_hyperparameters.py 64 8 512    # window=64, dilation=8, seq_len=512
```

Outputs:
- `plots/hyperparameter_comparison.png` - Attention patterns for 8 configurations
- `plots/sparsity_speedup_tradeoff.png` - Sparsity vs estimated speedup
- `output/hyperparameter_analysis.txt` - Detailed analysis

## Implementation Details

### Architecture

**Baseline Model (`GPTModel` in `gpt.py`):**
- Standard full self-attention: O(n²) complexity
- Each token attends to all previous tokens

**Sparse Model (`GPTWindowModel` in `gpt_win.py`):**
- Sliding-window + dilated attention: O(n·w) complexity
- Inherits from `GPTModel`, replacing only the attention layers
- Window size: w=96 tokens
- Dilation: d=4 (every 4th token beyond window)

### Code Reuse Through Inheritance

`gpt_win.py` demonstrates clean inheritance patterns:

```python
class GPTWindowModel(GPTModel):
    """Inherits embeddings, forward pass, output layer from GPTModel."""
    def __init__(self, ...):
        super().__init__(...)  # Initialize base GPTModel
        # Replace attention layers with windowed versions
        self.layers = ModuleList([TransformerDecoderBlockWindow(...) for _ in range(layers)])

class TransformerDecoderBlockWindow(TransformerDecoderBlock):
    """Inherits LayerNorm, FFN from base. Replaces only MHA."""
    def __init__(self, ...):
        super().__init__(...)
        self.mha = CustomWindowMHA(...)  # Override attention
```

### Attention Pattern

Each token at position i attends to:
- **Local window:** Previous w=96 tokens
- **Dilated links:** Every d=4th token before the window

## Experimental Design

### Controls for Rigor

1. **Randomized Order:** Baseline/Window training order randomized per run
2. **GPU Cleanup:** Aggressive cache clearing between runs
3. **Kernel Warmup:** 3 warmup iterations before timing
4. **Multiple Runs:** 10 runs for statistical confidence

### Statistical Validation

- Two-sample t-tests for training time comparison
- Cohen's d effect size calculation
- Confidence intervals (mean ± std)
- p-value significance testing (α = 0.05)

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM:** 16GB+ recommended
- **Runtime:** ~5 hours for full 10-run experiment on RTX-class GPU

## References

1. Vaswani et al., "Attention Is All You Need," 2017
2. Child et al., "Generating Long Sequences with Sparse Transformers," 2019
3. Beltagy et al., "Longformer: The Long-Document Transformer," 2020
4. Zaheer et al., "Big Bird: Transformers for Longer Sequences," 2021
5. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention," 2022

## License

This project is for educational purposes as part of EN.705.743 at Johns Hopkins University.
