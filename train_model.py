import torch
import numpy as np
import matplotlib.pyplot as plt
from gpt import GPTModel

import math, time
from tqdm import tqdm

# since we didn't really cover how to do this in lecture
# this creates a learning rate schedule for you. Refer to the
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step.
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum) / float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum - warmup_steps
            tail_steps = total_steps - warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592 * prog) + 1.0) * 0.5) * 0.9 + 0.1
        return max(lrmult, 0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler


# ===========================================================================

"""
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
"""

def train():

    # Load dataset
    print("\nLoading training data...")
    data = np.load("training_data.npy")
    print(f"Loaded data shape: {data.shape}")

    # Split into inputs and targets
    inputs = data[:, :-1]   # All tokens except the last
    targets = data[:, 1:]   # All tokens except the first

    # Convert to PyTorch tensors
    inputs = torch.from_numpy(inputs).long()
    targets = torch.from_numpy(targets).long()

    N, S = inputs.shape
    print(f"Dataset size: {N} sequences, seq_len={S}")

    vocab_size = int(data.max()) + 1
    print(f"Inferred vocab size: {vocab_size}")

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    model = GPTModel(
        d_model=512,
        n_heads=16,
        layers=8,
        vocab_size=vocab_size,
        max_seq_len=S,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    batch_size = 8
    learning_rate = 3e-4
    warmup_steps = 500
    print(f"Batch size: {batch_size} | Learning rate: {learning_rate} | Warmup steps: {warmup_steps}")
    steps_per_epoch = math.ceil(N / batch_size)
    total_steps = steps_per_epoch
    print(f"Total steps: {total_steps} | Steps per epoch: {steps_per_epoch}")


    # Optimizer, scheduler, loss
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps)
    loss_fn = torch.nn.CrossEntropyLoss()


    # Training loop (manual batching)
    print("\nStarting training...")
    model.train()

    losses, tokens_seen = [], []
    total_tokens = 0
    step = 0

    # Shuffle once (only one epoch)
    perm = torch.randperm(N)
    inputs = inputs[perm]
    targets = targets[perm]

    progress = tqdm(range(0, N, batch_size), desc="Training", dynamic_ncols=True)
    start_training = time.time()

    for start in progress:
        end = min(start + batch_size, N)
        batch_inputs = inputs[start:end].to(device, non_blocking=True)
        batch_targets = targets[start:end].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(batch_inputs)  # (B, S, V)
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

        if (step % 500 == 0) or (step == total_steps):
            plt.figure(figsize=(10, 6))
            plt.plot(tokens_seen, losses, alpha=0.35, label='Raw')
            if len(losses) > 20:
                window = min(20, max(2, len(losses) // 5))
                sm = np.convolve(losses, np.ones(window) / window, mode='valid')
                sm_tokens = tokens_seen[window - 1:window - 1 + len(sm)]
                plt.plot(sm_tokens, sm, linewidth=2, label='Smoothed')
            plt.xlabel('Total Tokens')
            plt.ylabel('Loss')
            plt.title('Training Loss vs Tokens')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('loss_curve.png')
            plt.close()

            # append loss info directly to tqdm bar
            progress.set_postfix({
                "step": f"{step}",
                "loss": f"{loss.item():.4f}" if loss is not None else "N/A"
            })

    # Save artifacts
    torch.save(model.state_dict(), "weights/model_weights.pt")
    np.save("data/npy/training_losses.npy", np.array(losses))
    np.save("tokens_seen.npy", np.array(tokens_seen))
    print("Saved model_weights.pt, training_losses.npy, tokens_seen.npy")

    # Final plots & saves
    if len(losses) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(tokens_seen, losses, alpha=0.35, label='Raw')
        # Apply a simple moving average to smooth the loss curve for visualization.
        # If we have more than 20 recorded losses, use a window size equal to 1/10 of
        # the total number of points (capped at 50, but at least 2). This helps reduce
        # short-term noise in the training loss, making overall convergence trends easier to see.
        # The smoothed values are plotted alongside the raw losses for comparison.
        if len(losses) > 20:
            window = min(50, max(2, len(losses) // 10))
            sm = np.convolve(losses, np.ones(window) / window, mode='valid')
            sm_tokens = tokens_seen[window - 1:window - 1 + len(sm)]
            plt.plot(sm_tokens, sm, linewidth=2, label='Smoothed')
    plt.xlabel('Total Tokens')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Tokens')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    print("Saved loss_curve.png")

    print(f"\nTraining completed! Steps={step}, Tokens={total_tokens:,}")
    if len(losses) > 0:
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Avg loss (last 100 recs): {np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses):.4f}")

    print(f"Training finished in {(time.time() - start_training) / 60:.2f} min")


if __name__ == "__main__":
    train()
