#!/usr/bin/env python3
"""
Exercise 3.1: Model Architecture & Parameter Counting

Learn how to:
- Create models of different sizes
- Count parameters in a model
- Understand the relationship between architecture and size
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import mlx.core as mx

# Create a tiny model
config = GPTConfig(
    sequence_len=256,
    vocab_size=65536,
    n_layer=2,        # Only 2 layers
    n_head=4,         # 4 attention heads
    n_kv_head=4,
    n_embd=128        # 128-dimensional embeddings
)

model = GPT(config)
model.init_weights()

print("=" * 60)
print("Model Configuration")
print("=" * 60)
print(f"  Layers: {config.n_layer}")
print(f"  Attention heads: {config.n_head}")
print(f"  Embedding dimension: {config.n_embd}")
print(f"  Vocabulary size: {config.vocab_size}")
print(f"  Max sequence length: {config.sequence_len}")

# Count parameters
def count_params(tree):
    """Recursively count all parameters in model"""
    total = 0
    if isinstance(tree, dict):
        for v in tree.values():
            total += count_params(v)
    elif isinstance(tree, list):
        for v in tree:
            total += count_params(v)
    elif hasattr(tree, 'size'):
        total += tree.size
    return total

nparams = count_params(model.parameters())
print(f"\nTotal parameters: {nparams:,} ({nparams/1e6:.2f}M)")

print("\n" + "=" * 60)
print("Explore Different Architectures")
print("=" * 60)

# Try different configurations
configs = [
    ("Tiny (d2)", dict(n_layer=2, n_head=4, n_embd=128)),
    ("Small (d4)", dict(n_layer=4, n_head=4, n_embd=256)),
    ("Medium (d6)", dict(n_layer=6, n_head=6, n_embd=384)),
    ("Large (d10)", dict(n_layer=10, n_head=8, n_embd=512)),
]

print("\nParameter count for different model sizes:\n")
for name, arch_params in configs:
    config = GPTConfig(
        sequence_len=256,
        vocab_size=65536,
        **arch_params
    )
    model = GPT(config)
    model.init_weights()
    params = count_params(model.parameters())
    print(f"{name:20s}: {params:12,} params ({params/1e6:6.2f}M)")

print("\n" + "=" * 60)
print("Understanding Parameter Growth")
print("=" * 60)

print("""
Key insights:

1. EMBEDDING LAYERS:
   - Token embedding: vocab_size × n_embd
   - Position embedding: sequence_len × n_embd

2. TRANSFORMER BLOCKS (per block):
   - Layer norm: 2 × n_embd
   - Attention:
     - QKV projection: 3 × n_embd × n_embd
     - Output projection: n_embd × n_embd
   - Feed-forward:
     - First layer: n_embd × (4 × n_embd)
     - Second layer: (4 × n_embd) × n_embd

3. OUTPUT HEAD:
   - Final layer norm: n_embd
   - Projection to vocab: n_embd × vocab_size

4. PARAMETER DOUBLING EFFECTS:
   - Doubling n_embd → 4× parameters (quadratic!)
   - Doubling n_layer → 2× parameters (linear)
   - Doubling n_head → 2× parameters (if n_embd increases proportionally)
""")

print("\n" + "=" * 60)
print("TODO: Try modifying the architecture")
print("=" * 60)
print("""
1. Change n_layer, n_head, n_embd and see how parameters change
2. Calculate: Why does doubling n_embd more than double parameters?
3. Find a configuration with ~50M parameters
4. Estimate training time for each size (rule of thumb:
   ~1 hour per 1B tokens per 10M params on Apple Silicon)
""")
