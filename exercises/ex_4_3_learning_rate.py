#!/usr/bin/env python3
"""
Exercise 4.3: Learning Rate Impact - Find the Sweet Spot

Learn how to:
- Train models with different learning rates
- Compare convergence behavior
- Understand the learning rate tradeoff
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import mlx.nn as nn

def train_with_lr(learning_rate, num_steps=100):
    """Train a tiny model with specified learning rate"""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64
    )
    model = GPT(config)
    model.init_weights()

    train_data = mx.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=mx.int32)
    optimizer = AdamW(learning_rate=learning_rate)

    def loss_fn(model, x, y):
        logits = model(x)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y.reshape(-1)
        log_probs = nn.log_softmax(logits_flat, axis=-1)
        losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)
        return mx.mean(losses)

    losses = []
    for step in range(num_steps):
        inputs = train_data[:, :-1]
        targets = train_data[:, 1:]
        loss_val, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        mx.eval(loss_val)
        losses.append(loss_val.item())

    return losses

print("=" * 60)
print("Comparing Different Learning Rates")
print("=" * 60)

# Try different learning rates
learning_rates = [0.00001, 0.0001, 0.001, 0.01]

print(f"\nTraining for {100} steps with different learning rates:\n")

all_results = {}
for lr in learning_rates:
    print(f"Training with LR = {lr:.5f}...", end=" ", flush=True)
    losses = train_with_lr(lr, num_steps=100)
    all_results[lr] = losses
    print(f"✓")

print(f"\n{'='*60}")
print("Results Summary")
print(f"{'='*60}\n")

print(f"{'Learning Rate':>15s} | {'Initial Loss':>12s} | {'Final Loss':>12s} | {'Improvement':>12s} | Status")
print(f"{'-'*80}")

for lr in learning_rates:
    losses = all_results[lr]
    initial = losses[0]
    final = losses[-1]
    improvement = initial - final

    # Status
    if final > 10 or any(l > 100 for l in losses[-5:]):
        status = "⚠️ DIVERGED (LR too high!)"
    elif final < 0.1:
        status = "✓ Excellent!"
    elif final < 1.0:
        status = "✓ Good"
    else:
        status = "~ Moderate"

    print(f"{lr:15.5f} | {initial:12.4f} | {final:12.4f} | {improvement:12.4f} | {status}")

print(f"\n{'='*60}")
print("Loss Curves Over Time")
print(f"{'='*60}\n")

# Simple ASCII plot
print("Step-by-step loss for first 20 steps:\n")
print(f"{'Step':>5s} |", end="")
for lr in learning_rates:
    print(f" LR={lr:.0e}  |", end="")
print()
print("-" * (5 + len(learning_rates) * 15))

for step in range(0, 20, 2):
    print(f"{step:5d} |", end="")
    for lr in learning_rates:
        loss_val = all_results[lr][step]
        # Scale loss for visualization (0-4 range maps to bar length)
        bar_len = min(int(loss_val / 4 * 10), 10)
        bar = "█" * bar_len + " " * (10 - bar_len)
        print(f" {bar}  |", end="")
    print()

print(f"\n{'='*60}")
print("Analysis")
print(f"{'='*60}")

# Find best learning rate
final_losses = {lr: all_results[lr][-1] for lr in learning_rates}
best_lr = min(final_losses, key=final_losses.get)
worst_lr = max(final_losses, key=final_losses.get)

print(f"\nBest learning rate: {best_lr:.5f} (final loss: {final_losses[best_lr]:.4f})")
print(f"Worst learning rate: {worst_lr:.5f} (final loss: {final_losses[worst_lr]:.4f})")

print(f"""
Key Observations:

1. TOO LOW (0.00001, 0.0001):
   - Loss decreases very slowly
   - Training takes many steps
   - Will eventually work but is inefficient

2. JUST RIGHT (0.001, 0.01):
   - Steady decrease in loss
   - Converges in reasonable number of steps
   - Sweet spot for this model

3. TOO HIGH (0.1+):
   - Loss oscillates or increases
   - May diverge (become NaN or Inf)
   - Model parameters become unstable

Learning Rate Heuristics:
- Start with 10^-3 to 10^-4
- If loss increases: LR is too high
- If loss barely decreases: LR is too low
- Good LR shows steady smooth decrease
- Different models need different LR!
  * Tiny models: higher LR works
  * Large models: often need lower LR
  * Fine-tuning: usually needs much lower LR than pretraining
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")
print("""
1. Try LR = 0.1 and 0.5 - do they diverge?
2. Plot losses more finely (every step instead of every 2)
3. Run for more steps - where does each LR plateau?
4. Try custom learning rates between 0.0001 and 0.01
5. For the best LR, how many steps until loss < 0.1?
6. Research: Why is learning rate so critical in neural networks?
""")
