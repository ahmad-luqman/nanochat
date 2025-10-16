#!/usr/bin/env python3
"""
Exercise 5.2: Basic Generation - Implementing Text Generation

Learn how to:
- Implement a simple generation loop
- Tokenize input and decode output
- Generate text from a model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
import mlx.core as mx
import mlx.nn as nn
import os

print("=" * 60)
print("Basic Generation: Making the Model Talk")
print("=" * 60)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
print(f"✓ Tokenizer loaded")
print(f"  Vocab size: {tokenizer.vocab_size}")

# Create small untrained model (for demonstration)
print("\nCreating model...")
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
print(f"✓ Model created (untrained - will generate gibberish!)")

def generate(prompt_text, max_tokens=30, temperature=1.0):
    """Simple generation function"""
    print(f"\n{'='*60}")
    print(f"Generation with temperature={temperature}")
    print(f"{'='*60}")

    # Encode prompt
    tokens = tokenizer.encode(prompt_text)
    print(f"Prompt: '{prompt_text}'")
    print(f"Encoded tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"Decoded: '{tokenizer.decode(tokens)}'")

    print(f"\nGenerating {max_tokens} tokens:")
    print(f"  '{prompt_text}", end="", flush=True)

    # Keep track of tokens for display
    full_sequence = tokens.copy()

    for i in range(max_tokens):
        # Get model predictions
        input_array = mx.array([full_sequence], dtype=mx.int32)
        logits = model(input_array)

        # Get logits for last position
        next_logits = logits[0, -1, :] / temperature

        # Sample next token
        next_token = mx.random.categorical(next_logits)
        next_token_id = int(next_token.item())

        # Decode and print
        token_text = tokenizer.decode([next_token_id])
        print(token_text, end="", flush=True)

        # Add to sequence
        full_sequence.append(next_token_id)

    print("'")
    print(f"\nFull generated text ({len(full_sequence)} tokens total)")

    return full_sequence

# Try generation with different temperatures
print(f"\n{'='*60}")
print("Generating Text")
print(f"{'='*60}")

prompt = "Once upon a time"
print(f"\nPrompt: '{prompt}'")
print(f"(Note: Model is untrained, output will be gibberish!)")

# Generate with different temperatures
generate(prompt, max_tokens=20, temperature=0.5)
generate(prompt, max_tokens=20, temperature=1.0)
generate(prompt, max_tokens=20, temperature=1.5)

print(f"\n{'='*60}")
print("Generation Loop Explained")
print(f"{'='*60}")

print("""
The generation loop:

1. INITIALIZE:
   - Start with prompt tokens
   - Keep track of full sequence

2. REPEAT N times:
   a) RUN MODEL on full sequence
      - Get logits for all positions
      - We only care about last position!

   b) APPLY TEMPERATURE:
      - logits_scaled = logits[-1, :] / temperature
      - Lower T = more confident (sharper distribution)
      - Higher T = more random (flatter distribution)

   c) SAMPLE from distribution:
      - Don't always pick argmax!
      - Use categorical distribution
      - Different token each time (usually)

   d) APPEND to sequence:
      - Add selected token to full_sequence
      - Keep growing the sequence

3. RETURN:
   - Full sequence of generated tokens
   - Decode back to text
   - Display to user

Why not use argmax (greedy)?
  - argmax = always pick highest probability
  - Result: repetitive, boring text
  - Real generation uses sampling
  - But temperature controls risk level
""")

print(f"\n{'='*60}")
print("With a Trained Model")
print(f"{'='*60}")

print("""
With a TRAINED model, you would see:
- Coherent text that follows language patterns
- Varied but sensible completions
- Different outputs for different temperatures

Example with trained d10 model (low temperature):
  Prompt: "Once upon a time"
  Generated: "once upon a time there was a small kingdom..."

Example with trained d10 model (high temperature):
  Prompt: "Once upon a time"
  Generated: "once upon a time, the purple elephant..."

The difference?
- Training teaches the model what comes after what
- Untrained model = random token sequence
- Trained model = follows patterns from training data
""")

print(f"\n{'='*60}")
print("Improvements to Try")
print(f"{'='*60}")

print("""
1. TOP-K FILTERING:
   - Only consider top K most likely tokens
   - Avoids very unlikely tokens
   - Improves quality

2. TOP-P (NUCLEUS) FILTERING:
   - Consider tokens that sum to top P probability
   - Adaptive to different distributions
   - More sophisticated

3. REPETITION PENALTY:
   - Reduce probability of recently used tokens
   - Avoid repetitive text
   - Common in language models

4. STOPPING TOKENS:
   - Stop generation at end-of-sequence token
   - Stop at punctuation
   - Stop after certain patterns

5. LENGTH PENALTY:
   - Prefer shorter or longer outputs
   - Common in machine translation
   - Control output verbosity
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Generate from different prompts
2. Generate very long outputs (100+ tokens)
3. See how temperature affects length/diversity
4. Implement top-k filtering
5. Add a stopping condition (e.g., stop at period)
6. Run with a trained checkpoint for real text
7. Compare quality of generated text at different temps
8. Try prompts in different styles/domains
""")
