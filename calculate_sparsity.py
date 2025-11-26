#!/usr/bin/env python3
"""
Calculate exact attended positions for sliding window + dilated attention.
"""


def count_attended_positions(window_size, dilation, seq_len):
    """
    Count exactly how many positions are attended in the sparse mask.
    """
    total_attended = 0

    for i in range(seq_len):
        # Local window: [max(0, i - window_size + 1), i]
        window_start = max(0, i - window_size + 1)
        window_count = i - window_start + 1

        # Dilated positions: every dilation-th position before window
        dilated_count = 0
        dilated_pos = i - window_size - dilation
        while dilated_pos >= 0:
            dilated_count += 1
            dilated_pos -= dilation

        total_attended += window_count + dilated_count

    return total_attended


# Calculate for your actual configuration
w, d, n = 96, 4, 256

attended = count_attended_positions(w, d, n)
causal_total = sum(range(1, n + 1))  # n(n+1)/2

density = attended / causal_total * 100
sparsity = 100 - density

print("=" * 60)
print(f"Configuration: window_size={w}, dilation={d}, seq_len={n}")
print("=" * 60)
print(f"Attended positions:  {attended:,}")
print(f"Causal total:        {causal_total:,}")
print(f"Density:             {density:.2f}%")
print(f"Sparsity:            {sparsity:.2f}%")
print()

# Show breakdown for a few positions
print("Example breakdown:")
print("-" * 60)
for i in [0, 50, 100, 150, 200, 255]:
    window_start = max(0, i - w + 1)
    window_count = i - window_start + 1

    dilated_count = 0
    dilated_pos = i - w - d
    while dilated_pos >= 0:
        dilated_count += 1
        dilated_pos -= d

    total = window_count + dilated_count
    print(f"Position {i:3d}: window={window_count:3d} + dilated={dilated_count:2d} = {total:3d} positions")