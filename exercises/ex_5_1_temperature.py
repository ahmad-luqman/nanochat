#!/usr/bin/env python3
"""
Exercise 5.1: Temperature - Control Randomness in Sampling

Learn how to:
- Understand temperature parameter
- See how it affects probability distributions
- Find the right balance between creativity and coherence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

print("=" * 60)
print("Temperature: Controlling Generation Randomness")
print("=" * 60)

# Simulate model predictions - 10 possible tokens
logits = mx.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

print(f"\nOriginal logits (model predictions):")
print(f"  {logits.tolist()}")
print(f"\nWithout temperature adjustment, top token has highest logit: {mx.argmax(logits).item()}")

# Look at probabilities without temperature (temperature = 1.0)
probs_default = nn.softmax(logits)
print(f"\nDefault probabilities (temperature=1.0):")
for i, p in enumerate(probs_default.tolist()):
    bar = "█" * int(p * 40)
    print(f"  Token {i}: {p:.4f} {bar}")

print(f"\n{'='*60}")
print("Effect of Temperature on Probabilities")
print(f"{'='*60}\n")

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temperatures:
    print(f"Temperature = {temp}:")
    print(f"{'':3s}  (lower = more confident, higher = more random)")

    # Apply temperature: divide logits by temperature
    scaled_logits = logits / temp
    probs = nn.softmax(scaled_logits)

    # Show top 3
    top_3_indices = mx.argsort(-probs)[:3]
    print(f"  Top 3 tokens: {top_3_indices.tolist()}")

    # Show probabilities
    for i in range(3):
        idx = top_3_indices[i]
        p = probs[idx]
        print(f"    Token {idx.item():d}: {p.item():.4f}")

    # Sample 10 times
    samples = []
    for _ in range(10):
        token = mx.random.categorical(scaled_logits)
        samples.append(token.item())

    unique = len(set(samples))
    print(f"  10 samples: {samples}")
    print(f"  Unique tokens: {unique}/10")
    print()

print(f"{'='*60}")
print("Visual Comparison")
print(f"{'='*60}\n")

print("Token probabilities for different temperatures:")
print()

for i in range(10):
    print(f"Token {i}: ", end="")
    for temp in temperatures:
        scaled_logits = logits / temp
        probs = nn.softmax(scaled_logits)
        p = probs[i].item()
        bar_len = int(p * 10)
        bar = "█" * bar_len + "-" * (10 - bar_len)
        print(f"{bar} ", end="")
    print()

print(f"\n  T={temperatures[0]} | T={temperatures[1]} | T={temperatures[2]} | T={temperatures[3]} | T={temperatures[4]}")

print(f"\n{'='*60}")
print("Real-world Examples")
print(f"{'='*60}")

print("""
Temperature = 0.1 (Very Deterministic):
  - Always picks the same token
  - Use for: factual Q&A, code generation
  - Risk: boring, repetitive

Temperature = 0.5-0.8 (Controlled Randomness):
  - Mostly picks top tokens, occasionally variations
  - Use for: normal conversation, creative writing
  - Best for: most applications

Temperature = 1.0 (Default):
  - Natural distribution from model
  - Use for: baseline behavior
  - May be too random or too boring depending on model

Temperature = 1.5+ (High Creativity):
  - Explores many different tokens
  - Use for: creative writing, brainstorming
  - Risk: nonsensical, incoherent text

Temperature > 2.0 (Extreme Randomness):
  - Almost uniformly random
  - Use for: never (usually)
  - Result: gibberish
""")

print(f"\n{'='*60}")
print("Mathematical Understanding")
print(f"{'='*60}")

print("""
Formula: P(token) = exp(logit / T) / sum(exp(logits / T))

When T → 0 (very low):
  - exp(logit / T) → exp(infinity) for highest logit
  - → exp(-infinity) for others
  - Result: one token gets ~100% probability (greedy)

When T = 1 (normal):
  - Standard softmax
  - Natural model predictions

When T → ∞ (very high):
  - All exponentials → exp(0) = 1
  - All tokens get equal probability
  - Result: uniform random (like dice roll)

Finding the right T:
  - T = 0.1-0.5: Factual, deterministic
  - T = 0.7-1.0: Balanced (recommended default)
  - T = 1.0-1.5: Creative, exploratory
  - T > 2.0: Usually too random
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")
print("""
1. Generate 10 samples with T=0.1 - notice they're all the same
2. Generate 10 samples with T=2.0 - notice the variety
3. For different temperatures, what percentage of the top 5 tokens?
4. What temperature gives you 50/50 chance of top 2 tokens?
5. Use a trained model and generate with different temps
6. For chat: what temperature do users prefer?
7. Compare temperatures 0.7, 0.8, 0.9 for fine-grained control
""")
