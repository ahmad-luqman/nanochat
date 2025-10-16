#!/usr/bin/env python3
"""
Exercise 4.1: Understanding Loss Functions

Learn how to:
- Understand cross-entropy loss
- Interpret loss values
- See how loss changes with predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

print("=" * 60)
print("Cross-Entropy Loss: How Wrong Is Our Prediction?")
print("=" * 60)

# Imagine we're predicting the next token
vocab_size = 10  # Simplified to 10 tokens
predicted_logits = mx.array([2.0, 1.0, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1])
true_token_id = 3  # The correct answer is token 3

print(f"\nPredicted logits: {predicted_logits.tolist()}")
print(f"True token ID: {true_token_id} (logit value: {predicted_logits[true_token_id].item():.2f})")

# Convert logits to probabilities
probs = nn.softmax(predicted_logits)
print(f"\nProbabilities (sum to 1.0):")
for i, p in enumerate(probs.tolist()):
    marker = " <-- TRUE LABEL" if i == true_token_id else ""
    bar = "█" * int(p * 50)
    print(f"  Token {i}: {p:.4f} {bar}{marker}")

# Cross-entropy loss
log_probs = nn.log_softmax(predicted_logits)
loss = -log_probs[true_token_id]
print(f"\nCross-Entropy Loss: {loss.item():.4f}")
print(f"  (Lower is better. Perfect prediction → 0)")

# Probability of true token
true_prob = probs[true_token_id].item()
print(f"\nProbability assigned to true token: {true_prob:.4f} ({true_prob*100:.2f}%)")
print(f"  This is good! We predicted token 3 with {true_prob*100:.1f}% confidence")

# Now let's see extreme cases
print(f"\n" + "=" * 60)
print("EXTREME CASES")
print("=" * 60)

# Perfect prediction
print(f"\n1. PERFECT PREDICTION:")
perfect_logits = mx.array([-100., -100., -100., 100., -100., -100., -100., -100., -100., -100.])
perfect_probs = nn.softmax(perfect_logits)
perfect_loss = -nn.log_softmax(perfect_logits)[true_token_id]
print(f"   Logits: [very negative..., very positive at pos 3, very negative...]")
print(f"   Probability of true token: {perfect_probs[true_token_id].item():.6f} (99.99%+)")
print(f"   Loss: {perfect_loss.item():.6f} (nearly 0 - excellent!)")

# Terrible prediction
print(f"\n2. TERRIBLE PREDICTION:")
terrible_logits = mx.array([100., 100., 100., -100., 100., 100., 100., 100., 100., 100.])
terrible_probs = nn.softmax(terrible_logits)
terrible_loss = -nn.log_softmax(terrible_logits)[true_token_id]
print(f"   Logits: [very positive..., very negative at pos 3, very positive...]")
print(f"   Probability of true token: {terrible_probs[true_token_id].item():.6f} (0.01%-)")
print(f"   Loss: {terrible_loss.item():.4f} (very high - terrible!)")

# Medium prediction
print(f"\n3. MEDIUM PREDICTION:")
medium_logits = mx.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
medium_probs = nn.softmax(medium_logits)
medium_loss = -nn.log_softmax(medium_logits)[true_token_id]
print(f"   Logits: [all equal]")
print(f"   Probability of true token: {medium_probs[true_token_id].item():.4f} (10% - random!)")
print(f"   Loss: {medium_loss.item():.4f}")

print(f"\n" + "=" * 60)
print("UNDERSTANDING LOSS VALUES")
print("=" * 60)
print(f"""
Loss Interpretation:
  - Loss ≈ 0.0   → Perfect prediction (100% confidence in correct token)
  - Loss ≈ 2.3   → Random guess (10% confidence for 10 tokens)
  - Loss ≈ 5.0   → Very wrong prediction
  - Loss ≈ 10+   → Extremely wrong prediction

Why log probabilities?
  - Log of probability is always negative
  - More negative = lower probability
  - Loss = -log_prob makes it positive and interpretable

Why negative sign?
  - We want to MINIMIZE loss
  - Lower loss = better predictions
  - Training adjusts weights to reduce this value
""")

print(f"\n" + "=" * 60)
print("BATCH LOSS")
print("=" * 60)

# Multiple predictions in a batch
batch_logits = mx.array([
    [3.0, 1.0, 0.5],  # Batch 1: should predict token 0
    [0.5, 2.5, 1.0],  # Batch 2: should predict token 1
    [0.2, 0.3, 3.0],  # Batch 3: should predict token 2
])
batch_targets = mx.array([0, 1, 2])

print(f"\nBatch of 3 predictions, vocab size 3")
print(f"Targets: {batch_targets.tolist()}")

# Compute loss for each
log_probs = nn.log_softmax(batch_logits, axis=-1)
batch_losses = -mx.take_along_axis(log_probs, batch_targets[:, None], axis=-1).squeeze(-1)

print(f"\nPer-sample losses: {[f'{l:.4f}' for l in batch_losses.tolist()]}")
print(f"Average loss: {mx.mean(batch_losses).item():.4f}")

print(f"\n" + "=" * 60)
print("TODO: Experiments")
print("=" * 60)
print("""
1. Change true_token_id and see how loss changes
2. Modify predicted_logits and watch the probabilities shift
3. What happens if all logits are the same?
4. What happens if one logit is extremely high?
5. Calculate loss for batch examples manually
6. Why is cross-entropy better than other loss functions for classification?
""")
