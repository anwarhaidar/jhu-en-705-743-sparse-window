import torch
import numpy as np
import gc
import random
import math
import time
import os
from tqdm import tqdm

from gpt import GPTModel
from gpt_win import GPTWindowModel
from train_model import cosine_with_warmup_lr_scheduler
from create_plots_improved import create_all_plots
from metrics_utils import save_metrics

def gpu_cleanup():
    """ Clean up GPU memory. """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection again
        gc.collect()
        time.sleep(2)  # Give GPU time to settle


def train_model(model, inputs, targets, device, batch_size=8, learning_rate=3e-4, 
                warmup_steps=500, model_name="Model"):
    """
    Train a model and return metrics: losses, tokens_seen, training_time, peak_memory
    """
    N, S = inputs.shape
    steps_per_epoch = math.ceil(N / batch_size)
    total_steps = steps_per_epoch

    # Optimizer, scheduler, loss
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining {model_name}...")
    model.train()

    losses, tokens_seen = [], []
    total_tokens = 0
    step = 0

    # Shuffle
    perm = torch.randperm(N)
    inputs_shuffled = inputs[perm]
    targets_shuffled = targets[perm]

    # Reset peak memory stats JUST before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup GPU (3 iterations to warm up kernels)
    print(f"Warming up GPU for {model_name}...")
    for i in range(3):
        batch_inputs = inputs_shuffled[:batch_size].to(device, non_blocking=True)
        batch_targets = targets_shuffled[:batch_size].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        B, S_cur, V = logits.shape
        loss = loss_fn(logits.reshape(-1, V), batch_targets.reshape(-1))
        loss.backward()
        opt.step()

    print(f"Starting timed training for {model_name}...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Reset stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    start_training = time.time()

    progress = tqdm(range(0, N, batch_size), desc=f"Training {model_name}", dynamic_ncols=True)

    for start in progress:
        end = min(start + batch_size, N)
        batch_inputs = inputs_shuffled[start:end].to(device, non_blocking=True)
        batch_targets = targets_shuffled[start:end].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        B, S_cur, V = logits.shape
        loss = loss_fn(logits.reshape(-1, V), batch_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        step += 1
        total_tokens += B * S_cur
        losses.append(loss.item())
        tokens_seen.append(total_tokens)

        progress.set_postfix({
            "step": f"{step}",
            "loss": f"{loss.item():.4f}"
        })

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    training_time = time.time() - start_training
    
    # Get peak memory usage
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    final_loss = losses[-1] if losses else float('inf')
    avg_loss_last_100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)

    print(f"\n{model_name} Training Complete!")
    print(f"  Total steps: {step}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"  Peak GPU memory: {peak_memory_mb:.2f} MB")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Avg loss (last 100): {avg_loss_last_100:.4f}")

    return {
        'losses': losses,
        'tokens_seen': tokens_seen,
        'training_time': training_time,
        'peak_memory_mb': peak_memory_mb,
        'final_loss': final_loss,
        'avg_loss_last_100': avg_loss_last_100,
        'total_steps': step,
        'total_tokens': total_tokens
    }

def compare_models_improved(num_runs=10, randomize_order=True):
    """
    Improved comparison with multiple runs and order randomization.
    """
    print("="*80)
    print(f"GPT MODEL COMPARISON - {num_runs} RUNS")
    print("Multiple runs with GPU cleanup and order randomization")
    print("="*80)
    
    # Start overall timing
    overall_start = time.time()

    # Load dataset
    print("\nLoading training data...")
    data = np.load("training_data.npy")
    print(f"Loaded data shape: {data.shape}")

    # Split into inputs and targets
    inputs = data[:, :-1]
    targets = data[:, 1:]

    # Convert to PyTorch tensors
    inputs = torch.from_numpy(inputs).long()
    targets = torch.from_numpy(targets).long()

    N, S = inputs.shape
    print(f"Dataset size: {N} sequences, seq_len={S}")

    vocab_size = int(data.max()) + 1
    print(f"Inferred vocab size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model hyperparameters
    d_model = 512
    n_heads = 16
    layers = 8
    batch_size = 8
    learning_rate = 3e-4
    warmup_steps = 500

    # Window parameters
    window_size = 96
    dilation = 4

    print(f"\nModel Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  layers: {layers}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"\nWindow Model Parameters:")
    print(f"  window_size: {window_size} (reduced for more sparsity)")
    print(f"  dilation: {dilation}")
    print(f"\nExperimental Setup:")
    print(f"  Number of runs: {num_runs}")
    print(f"  Randomize order: {randomize_order}")

    all_baseline_metrics = []
    all_window_metrics = []
    
    # Track timing for each run
    run_times = []

    for run in range(num_runs):
        run_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"RUN {run + 1} / {num_runs}")
        print(f"{'='*80}")
        
        # Randomize order to control for order effects
        if randomize_order:
            train_baseline_first = random.choice([True, False])
        else:
            train_baseline_first = True
        
        print(f"Order: {'Baseline → Window' if train_baseline_first else 'Window → Baseline'}")
        
        # Determine training order for this run
        training_order = ['baseline', 'window'] if train_baseline_first else ['window', 'baseline']
        
        metrics_this_run = {}
        
        for model_type in training_order:
            print(f"\n{'='*80}")
            print(f"TRAINING {model_type.upper()} MODEL (Run {run + 1})")
            print(f"{'='*80}")
            
            # Aggressive cleanup before creating model
            print("Performing aggressive GPU cleanup...")
            gpu_cleanup()
            
            # Create model
            if model_type == 'baseline':
                model = GPTModel(
                    d_model=d_model,
                    n_heads=n_heads,
                    layers=layers,
                    vocab_size=vocab_size,
                    max_seq_len=S,
                ).to(device)
                model_name = "Baseline"
            else:
                model = GPTWindowModel(
                    d_model=d_model,
                    n_heads=n_heads,
                    layers=layers,
                    vocab_size=vocab_size,
                    max_seq_len=S,
                    window_size=window_size,
                    dilation=dilation
                ).to(device)
                model_name = "Window"
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"{model_name} model has {param_count:,} parameters.")
            
            # Train
            metrics = train_model(
                model, inputs, targets, device,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                model_name=model_name
            )
            
            metrics_this_run[model_type] = metrics
            
            # Save the window model from the first run
            if model_type == 'window' and run == 0:
                torch.save(model.state_dict(), os.path.join("output", "window_model_weights.pt"))
                print("\nSaved window_model_weights.pt")
            
            # Aggressive cleanup after training
            print(f"\nCleaning up after {model_name} training...")
            del model
            gpu_cleanup()
        
        # Store metrics
        if 'baseline' in metrics_this_run:
            all_baseline_metrics.append(metrics_this_run['baseline'])
        if 'window' in metrics_this_run:
            all_window_metrics.append(metrics_this_run['window'])
        
        run_time = time.time() - run_start
        run_times.append(run_time)
        print(f"\n{'='*80}")
        print(f"RUN {run + 1} COMPLETED in {run_time:.2f}s ({run_time/60:.2f} min)")
        print(f"{'='*80}")

    # Calculate overall time
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"\nTotal experiment time: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print(f"Average time per run: {np.mean(run_times):.2f}s ({np.mean(run_times)/60:.2f} min)")
    print(f"Run time range: {np.min(run_times):.2f}s - {np.max(run_times):.2f}s")

    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
    print("="*80)
    
    baseline_times = [m['training_time'] for m in all_baseline_metrics]
    window_times = [m['training_time'] for m in all_window_metrics]
    
    baseline_memories = [m['peak_memory_mb'] for m in all_baseline_metrics]
    window_memories = [m['peak_memory_mb'] for m in all_window_metrics]
    
    baseline_losses = [m['avg_loss_last_100'] for m in all_baseline_metrics]
    window_losses = [m['avg_loss_last_100'] for m in all_window_metrics]

    # Print header
    print(f"\n{'Metric':<30} {'Baseline':<25} {'Window':<25} {'Improvement':<20}")
    print("-" * 100)

    # Training time
    baseline_time_mean = np.mean(baseline_times)
    baseline_time_std = np.std(baseline_times)
    window_time_mean = np.mean(window_times)
    window_time_std = np.std(window_times)
    time_improvement = (baseline_time_mean - window_time_mean) / baseline_time_mean * 100

    baseline_time_str = f"{baseline_time_mean:>7.2f} ± {baseline_time_std:<5.2f}"
    window_time_str = f"{window_time_mean:>7.2f} ± {window_time_std:<5.2f}"
    print(f"{'Training Time (s)':<30} {baseline_time_str:<25} {window_time_str:<25} {time_improvement:>+19.2f}%")

    # Memory
    baseline_mem_mean = np.mean(baseline_memories)
    baseline_mem_std = np.std(baseline_memories)
    window_mem_mean = np.mean(window_memories)
    window_mem_std = np.std(window_memories)
    memory_improvement = (baseline_mem_mean - window_mem_mean) / baseline_mem_mean * 100

    baseline_mem_str = f"{baseline_mem_mean:>7.2f} ± {baseline_mem_std:<5.2f}"
    window_mem_str = f"{window_mem_mean:>7.2f} ± {window_mem_std:<5.2f}"
    print(f"{'Peak GPU Memory (MB)':<30} {baseline_mem_str:<25} {window_mem_str:<25} {memory_improvement:>+19.2f}%")

    # Loss
    baseline_loss_mean = np.mean(baseline_losses)
    baseline_loss_std = np.std(baseline_losses)
    window_loss_mean = np.mean(window_losses)
    window_loss_std = np.std(window_losses)
    loss_diff = (window_loss_mean - baseline_loss_mean) / baseline_loss_mean * 100

    baseline_loss_str = f"{baseline_loss_mean:>7.4f} ± {baseline_loss_std:<7.4f}"
    window_loss_str = f"{window_loss_mean:>7.4f} ± {window_loss_std:<7.4f}"
    print(f"{'Avg Loss (last 100)':<30} {baseline_loss_str:<25} {window_loss_str:<25} {loss_diff:>+19.2f}%")
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    if len(baseline_times) > 1:
        # Two-sample t-test for time
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(baseline_times, window_times)
        print(f"\nTraining Time Difference:")
        print(f"  t-statistic: {t_stat:.4f}")
        
        # Format p-value properly for very small values
        if p_value < 0.0001:
            print(f"  p-value: < 0.0001 (extremely significant)")
        else:
            print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Statistically significant at p < 0.05")
        else:
            print(f"  ✗ Not statistically significant")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(baseline_times) + np.var(window_times)) / 2)
        cohens_d = (baseline_time_mean - window_time_mean) / pooled_std
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print(f"\nWith {num_runs} run(s) and {'randomized' if randomize_order else 'fixed'} order:")
    print(f"  • Training speed improvement: {time_improvement:+.2f}%")
    print(f"  • Memory usage improvement: {memory_improvement:+.2f}%")
    print(f"  • Quality difference: {loss_diff:+.2f}%")
    
    if time_improvement > 5 and abs(loss_diff) < 5:
        print("\n SUCCESS: Window model is faster with comparable quality!")
    elif time_improvement > 0 and abs(loss_diff) < 5:
        print("\n Window model shows modest improvement with comparable quality")
    else:
        print("\n Results are mixed - consider more runs or parameter tuning")
    
    # Save results
    with open(os.path.join('output', 'improved_comparison_results.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"IMPROVED COMPARISON RESULTS ({num_runs} runs)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total experiment time: {overall_time:.2f}s ({overall_time/60:.2f} min)\n")
        f.write(f"Average time per run: {np.mean(run_times):.2f}s ({np.mean(run_times)/60:.2f} min)\n\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Order randomized: {randomize_order}\n\n")
        f.write(f"Training Time: {baseline_time_mean:.2f} ± {baseline_time_std:.2f}s → "
                f"{window_time_mean:.2f} ± {window_time_std:.2f}s ({time_improvement:+.2f}%)\n")
        f.write(f"Peak Memory: {baseline_mem_mean:.2f} ± {baseline_mem_std:.2f} MB → "
                f"{window_mem_mean:.2f} ± {window_mem_std:.2f} MB ({memory_improvement:+.2f}%)\n")
        f.write(f"Avg Loss: {baseline_loss_mean:.4f} ± {baseline_loss_std:.4f} → "
                f"{window_loss_mean:.4f} ± {window_loss_std:.4f} ({loss_diff:+.2f}%)\n")
    
    print("\nSaved improved_comparison_results.txt")
    
    # Save metrics for later use
    save_metrics(all_baseline_metrics, all_window_metrics, os.path.join('output','experiment_results.npz'))

    return all_baseline_metrics, all_window_metrics

if __name__ == "__main__":
    # Run with 10 iterations and randomized order
    baseline_metrics, window_metrics = compare_models_improved(num_runs=10, randomize_order=True)
    # Create all plots
    create_all_plots(baseline_metrics, window_metrics)
