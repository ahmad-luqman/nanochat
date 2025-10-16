#!/usr/bin/env python3
"""
Exercise 4.2: Tiny Training - Teaching a Model a Simple Pattern

Learn how to:
- Set up a training loop
- Compute loss and gradients
- Update weights with an optimizer
- Watch loss decrease as the model learns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import mlx.nn as nn

print("=" * 60)
print("Training a Tiny Model to Memorize a Pattern")
print("=" * 60)

# Tiny model - small enough to run quickly
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

print(f"\nModel created:")
print(f"  Layers: {config.n_layer}")
print(f"  Heads: {config.n_head}")
print(f"  Embedding dim: {config.n_embd}")
print(f"  Vocab size: {config.vocab_size}")

# Tiny dataset: just repeat the sequence [1, 2, 3, 4, 5]
# We'll train the model to memorize this pattern
train_data = mx.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=mx.int32)

print(f"\nTraining data pattern: [1, 2, 3, 4, 5] repeated twice")
print(f"  Given [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]")
print(f"  Predict [2, 3, 4, 5, 1, 2, 3, 4, 5, ?]")

# Optimizer
optimizer = AdamW(learning_rate=0.01)

# Loss function
def loss_fn(model, x, y):
    """Compute cross-entropy loss for next token prediction"""
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = y.reshape(-1)

    # Cross-entropy loss
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)

    return mx.mean(losses)

# Training loop
print(f"\n{'='*60}")
print("Training Loop")
print(f"{'='*60}\n")

print(f"{'Step':>5s} | {'Loss':>8s} | Status")
print(f"{'-'*40}")

for step in range(200):
    # Create input/target pairs
    inputs = train_data[:, :-1]   # [1, 2, 3, 4, 5, 1, 2, 3, 4]
    targets = train_data[:, 1:]   # [2, 3, 4, 5, 1, 2, 3, 4, 5]

    # Compute loss and gradients
    loss_val, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    mx.eval(loss_val)

    # Display progress
    if step % 20 == 0 or step == 199:
        loss_item = loss_val.item()
        # Simple status indicator
        if loss_item < 0.5:
            status = "✓ Excellent!"
        elif loss_item < 1.0:
            status = "✓ Good"
        elif loss_item < 2.0:
            status = "~ Improving"
        else:
            status = "~ Still learning"
        print(f"{step:5d} | {loss_item:8.4f} | {status}")

# Test the trained model
print(f"\n{'='*60}")
print("Testing: Can the model predict the next token?")
print(f"{'='*60}\n")

test_input = mx.array([[1, 2]], dtype=mx.int32)
logits = model(test_input)
next_token_logits = logits[0, -1, :]
predicted_token = mx.argmax(next_token_logits)
predicted_prob = nn.softmax(next_token_logits)[mx.argmax(next_token_logits)]

print(f"Given [1, 2], model predicts: {predicted_token.item()}")
print(f"Expected: 3")
print(f"Confidence: {predicted_prob.item()*100:.1f}%")

if predicted_token.item() == 3:
    print(f"✓ CORRECT! The model learned the pattern!")
else:
    print(f"✗ Wrong prediction. Keep training!")

# Test more
print(f"\nMore predictions:")
test_cases = [
    ([2, 3], 4),
    ([4, 5], 1),
    ([3, 4], 5),
    ([5, 1], 2),
]

correct = 0
for test_seq, expected in test_cases:
    test_input = mx.array([test_seq], dtype=mx.int32)
    logits = model(test_input)
    pred = mx.argmax(logits[0, -1, :]).item()
    match = "✓" if pred == expected else "✗"
    print(f"  {match} Given {test_seq}, predicted {pred} (expected {expected})")
    if pred == expected:
        correct += 1

print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct*100//len(test_cases)}%)")

print(f"\n{'='*60}")
print("KEY INSIGHTS")
print(f"{'='*60}")
print("""
1. TRAINING LOOP:
   - Forward pass: get predictions
   - Compute loss: how wrong are we?
   - Backward pass: compute gradients
   - Update weights: move in direction of less error

2. LEARNING RATE:
   - Too high (>0.1): loses diverge or bounce around
   - Too low (<0.0001): learning is very slow
   - Just right: steady decrease in loss

3. CONVERGENCE:
   - Loss should decrease smoothly
   - Plateaus mean the model has learned what it can
   - This simple pattern should converge quickly

4. GENERALIZATION vs MEMORIZATION:
   - This model is memorizing the exact pattern
   - On new unseen data, it would perform poorly
   - Real training uses much more diverse data
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")
print("""
1. Train for more steps and watch loss plateau
2. Try a more complex pattern like [1,2,3,4,5,6,7,8,9,10]
3. Try different learning rates (0.001, 0.1, 1.0)
4. What happens with learning rate too high?
5. Train on random data and compare loss behavior
6. How long does it take to converge on different patterns?
""")
