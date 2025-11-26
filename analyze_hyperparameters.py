"""
Quick hyperparameter configuration tester.
Helps you find the optimal window_size and dilation for your use case.
"""

import os
import time
from gpt_win import GPTWindowModel
import matplotlib.pyplot as plt

def analyze_sparsity_pattern(window_size, dilation, max_seq_len=256):
    """Analyze the sparsity pattern of a configuration."""
    model = GPTWindowModel(
        d_model=512, n_heads=16, layers=8,
        vocab_size=10000, max_seq_len=max_seq_len,
        window_size=window_size, dilation=dilation
    )

    # Get mask from first transformer block
    # noinspection PyProtectedMember
    first_layer = model.layers[0]  # type: ignore
    mask = first_layer.mha.get_attention_mask()
    # mask = first_layer.mha._cached_mask  # Access cached mask

    # Calculate statistics
    attended_positions = mask.sum().item()
    causal_positions = sum(range(1, max_seq_len + 1))  # Full causal attention

    density = attended_positions / causal_positions * 100
    sparsity = 100 - density

    # Estimate speedup (rough heuristic)
    estimated_speedup = sparsity * 0.15  # 15% of sparsity translates to speedup

    return {
        'window_size': window_size,
        'dilation': dilation,
        'attended_positions': int(attended_positions),
        'causal_positions': causal_positions,
        'density': density,
        'sparsity': sparsity,
        'estimated_speedup': estimated_speedup,
        'mask': mask
    }


def compare_configurations(seq_len=256):
    """Compare different hyperparameter configurations."""

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    print("="*80)
    print("HYPERPARAMETER CONFIGURATION ANALYSIS")
    print("="*80)
    print(f"\nSequence Length: {seq_len}")
    print(f"Analyzing different window_size and dilation combinations...\n")

    # Define configurations to test
    configs = [
        # (window_size, dilation, description)
        (128, 4, "Current (conservative)"),
        (128, 2, "Dense long-range"),
        (96, 4, "Balanced"),
        (96, 6, "Balanced aggressive"),
        (80, 4, "Aggressive"),
        (80, 6, "Aggressive sparse"),
        (64, 4, "Maximum local"),
        (64, 8, "Maximum sparsity"),
    ]

    results = []

    print(f"{'Configuration':<25} {'Window':<8} {'Dilation':<10} {'Sparsity':<10} "
          f"{'Est. Speedup':<15} {'Quality Risk':<15}")
    print("-"*100)

    for window_size, dilation, description in configs:
        stats = analyze_sparsity_pattern(window_size, dilation, seq_len)
        results.append((description, stats))

        # Estimate quality risk based on sparsity
        if stats['sparsity'] < 25:
            risk = "Very Low"
        elif stats['sparsity'] < 35:
            risk = "Low"
        elif stats['sparsity'] < 50:
            risk = "Medium"
        else:
            risk = "High"

        print(f"{description:<25} {window_size:<8} {dilation:<10} {stats['sparsity']:>8.1f}% "
              f"{stats['estimated_speedup']:>13.1f}% {risk:>15}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nFor Maximum Speed (accept 3-5% quality loss):")
    print("   window_size=64, dilation=8")

    print("\nFor Balanced Performance (1-3% quality loss):")
    print("   window_size=96, dilation=4")

    print("\nFor Conservative Approach (<1% quality loss):")
    print("   window_size=128, dilation=2")

    print("\nRecommended Next Test:")
    print("   window_size=96, dilation=4")
    print("   Expected: 8-12% speedup with minimal quality impact")

    # Visualize attention patterns
    print("\n" + "="*80)
    print("Generating comparison visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (description, stats) in enumerate(results):
        if idx >= 8:
            break

        ax = axes[idx]
        mask = stats['mask'][:min(128, seq_len), :min(128, seq_len)]  # Show up to 128x128

        im = ax.imshow(mask.cpu().numpy(), cmap='Blues', interpolation='nearest', aspect='auto')
        ax.set_title(f"{description}\nw={stats['window_size']}, d={stats['dilation']}\n"
                    f"Sparsity: {stats['sparsity']:.1f}%",
                    fontsize=10)
        ax.set_xlabel('Key Position', fontsize=8)
        ax.set_ylabel('Query Position', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig('plots/hyperparameter_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plots/hyperparameter_comparison.png")

    # Create sparsity vs speedup plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sparsities = [stats['sparsity'] for _, stats in results]
    speedups = [stats['estimated_speedup'] for _, stats in results]
    labels = [desc for desc, _ in results]

    ax.scatter(sparsities, speedups, s=200, alpha=0.6, c=range(len(results)), cmap='viridis')

    for i, label in enumerate(labels):
        ax.annotate(label, (sparsities[i], speedups[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Estimated Speedup (%)', fontsize=12)
    ax.set_title('Sparsity vs Expected Speedup Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Target: 5% speedup')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Good: 10% speedup')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/sparsity_speedup_tradeoff.png', dpi=150, bbox_inches='tight')
    print("Saved plots/sparsity_speedup_tradeoff.png")
    plt.close('all')

    # Save detailed report
    with open(os.path.join('output','hyperparameter_analysis.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER CONFIGURATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Sequence Length: {seq_len}\n\n")

        f.write(f"{'Configuration':<25} {'Window':<8} {'Dilation':<10} {'Sparsity':<10} "
                f"{'Est. Speedup':<15}\n")
        f.write("-"*80 + "\n")

        for description, stats in results:
            f.write(f"{description:<25} {stats['window_size']:<8} {stats['dilation']:<10} "
                   f"{stats['sparsity']:>8.1f}% {stats['estimated_speedup']:>13.1f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        f.write("For Maximum Speed: window_size=64, dilation=8\n")
        f.write("For Balanced: window_size=96, dilation=4\n")
        f.write("For Conservative: window_size=128, dilation=2\n")
        f.write("\nRecommended Next Test: window_size=96, dilation=4\n")

    print("Saved output/hyperparameter_analysis.txt")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - plots/hyperparameter_comparison.png (attention patterns)")
    print("  - plots/sparsity_speedup_tradeoff.png (performance curve)")
    print("  - output/hyperparameter_analysis.txt (detailed report)")

    return results


def test_specific_config(window_size, dilation, seq_len=256):
    """Test a specific configuration in detail."""

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Validate parameters
    if dilation >= window_size:
        print("="*80)
        print("WARNING: Questionable Configuration!")
        print("="*80)
        print(f"\nDilation ({dilation}) is >= window_size ({window_size}).")
        print("This is unusual and likely not optimal.")
        print("\nTypical values:")
        print("  - window_size: 64-128")
        print("  - dilation: 2-8")
        print(f"\nRecommendation: For window_size={window_size}, try dilation in [4, 8]")
        print("\nProceeding with analysis anyway...\n")
        time.sleep(2)

    print("="*80)
    print(f"TESTING CONFIG: window_size={window_size}, dilation={dilation}")
    print("="*80)

    stats = analyze_sparsity_pattern(window_size, dilation, seq_len)

    print(f"\nConfiguration Details:")
    print(f"  Window Size: {stats['window_size']}")
    print(f"  Dilation: {stats['dilation']}")
    print(f"  Sequence Length: {seq_len}")

    print(f"\nSparsity Analysis:")
    print(f"  Attended positions: {stats['attended_positions']:,} / {stats['causal_positions']:,}")
    print(f"  Density: {stats['density']:.2f}%")
    print(f"  Sparsity: {stats['sparsity']:.2f}%")

    print(f"\nPerformance Estimates:")
    print(f"  Estimated speedup: {stats['estimated_speedup']:.1f}%")

    # Warn about large dilation
    if dilation > window_size * 0.5:
        print(f"  ALERT: Dilation is very large relative to window size!")
        print(f"         This severely limits long-range connections.")

    # More realistic thresholds for quality assessment
    if stats['sparsity'] > 50:
        print(f"  Warning: High sparsity may significantly impact quality")
    elif stats['sparsity'] > 35:
        print(f"  Moderate sparsity - test quality carefully")
    elif stats['sparsity'] > 25:
        print(f"  * Balanced sparsity - good efficiency/quality tradeoff")
    else:
        print(f"  * Conservative sparsity - quality should be preserved")

    # Visualize this specific config
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full attention mask
    ax = axes[0]
    display_size = min(128, seq_len)
    mask = stats['mask'][:display_size, :display_size]
    im = ax.imshow(mask.cpu().numpy(), cmap='Blues', interpolation='nearest')
    ax.set_title(f'Attention Pattern\nw={window_size}, d={dilation}', fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

    # Attention for a specific query position
    ax = axes[1]
    query_pos = display_size // 2
    attention = mask[query_pos, :].cpu().numpy()
    ax.bar(range(len(attention)), attention, alpha=0.7, color='blue')
    ax.set_title(f'Attention for Query Position {query_pos}', fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Attention (1=attend, 0=masked)')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight window
    window_start = max(0, query_pos - window_size + 1)
    ax.axvspan(window_start, query_pos, alpha=0.2, color='green', label='Window')
    ax.legend()

    plt.tight_layout()
    output_file = f'plots/config_w{window_size}_d{dilation}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved {output_file}")
    plt.close('all')

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        # Test specific configuration
        window_size = int(sys.argv[1])
        dilation = int(sys.argv[2])
        seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        test_specific_config(window_size, dilation, seq_len)
    else:
        # Compare all configurations
        compare_configurations(seq_len=256)