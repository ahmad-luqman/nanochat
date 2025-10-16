#!/usr/bin/env python3
"""
Exercise 3.2: Forward Pass - Watch Data Flow Through Model

Learn how to:
- Create dummy inputs and process them
- Understand tensor shapes through the model
- Interpret model outputs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import mlx.core as mx

# Small model
config = GPTConfig(
    sequence_len=256,
    vocab_size=65536,
    n_layer=2,
    n_head=4,
    n_kv_head=4,
    n_embd=128
)
model = GPT(config)
model.init_weights()

print("=" * 60)
print("Forward Pass: Data Flow Through Model")
print("=" * 60)

# Create dummy input: batch of 1, sequence of 10 tokens
input_tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32)
print(f"\n1. INPUT TOKENS")
print(f"   Shape: {input_tokens.shape}")  # (1, 10)
print(f"   Batch size: 1, Sequence length: 10")
print(f"   Token IDs: {input_tokens.tolist()}")

# Forward pass
logits = model(input_tokens)
print(f"\n2. MODEL OUTPUT (LOGITS)")
print(f"   Shape: {logits.shape}")  # (1, 10, 65536)
print(f"   For each position, we have 65,536 scores (one per token)")

# Interpret the output - look at last position
print(f"\n3. LAST POSITION PREDICTIONS")
last_position_logits = logits[0, -1, :]  # Shape: (65536,)
print(f"   Last position logits shape: {last_position_logits.shape}")

# Get top 5 predicted tokens
top_5_indices = mx.argpartition(-last_position_logits, 5)[:5]
top_5_logits = mx.take(last_position_logits, top_5_indices)
print(f"\n   Top 5 predicted token IDs: {top_5_indices.tolist()}")
print(f"   Top 5 logit scores: {[f'{v:.2f}' for v in top_5_logits.tolist()]}")

# Show data flow for all positions
print(f"\n4. ALL POSITIONS OUTPUT")
print(f"   Position | Output Shape        | Top Token ID")
print(f"   " + "-" * 50)
for pos in range(input_tokens.shape[1]):
    pos_logits = logits[0, pos, :]
    top_token = mx.argmax(pos_logits)
    print(f"   {pos:8d} | (65536,) per position | {top_token.item():10d}")

# Demonstrate different batch sizes
print(f"\n" + "=" * 60)
print("EXPLORE: Different Batch Sizes and Sequence Lengths")
print("=" * 60)

test_cases = [
    ("Single short sequence", mx.array([[1, 2, 3]], dtype=mx.int32)),
    ("Single long sequence", mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=mx.int32)),
    ("Batch of 2, length 5", mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=mx.int32)),
    ("Batch of 4, length 8", mx.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ], dtype=mx.int32)),
]

print()
for name, test_input in test_cases:
    output = model(test_input)
    print(f"{name:30s} | Input: {test_input.shape} → Output: {output.shape}")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. INPUT SHAPE: (batch_size, sequence_length)
   - All tokens are processed in parallel (batch_size)
   - Each sequence has sequence_length tokens

2. OUTPUT SHAPE: (batch_size, sequence_length, vocab_size)
   - For each position, we get vocab_size scores
   - Highest score = most likely next token

3. INTERPRETATION:
   - logits[b, t, :] = prediction for position t in batch b
   - argmax(logits[b, t, :]) = predicted token
   - softmax(logits[b, t, :]) = probabilities

4. CAUSAL STRUCTURE:
   - Each position only looks at previous tokens
   - Position 0 uses only its own embedding
   - Position t uses tokens 0..t

5. EFFICIENCY:
   - Process whole batch at once (parallelization)
   - All positions computed simultaneously
   - No need to generate one token at a time during training
""")

print("\n" + "=" * 60)
print("TODO: Experiments")
print("=" * 60)
print("""
1. Try different sequence lengths (1, 10, 50, 256)
2. Try batch_size > 1
3. Look at logits for different positions
4. What happens if you feed very long sequences?
5. Compare predictions for different input values
""")
