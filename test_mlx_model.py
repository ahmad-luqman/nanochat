#!/usr/bin/env python3
"""
Quick test script to validate MLX model port
"""
import mlx.core as mx
from nanochat.gpt_mlx import GPT, GPTConfig

print("Testing MLX GPT model...")

# Create a tiny model config
config = GPTConfig(
    sequence_len=128,
    vocab_size=1000,
    n_layer=2,
    n_head=4,
    n_kv_head=4,
    n_embd=128
)

print(f"Config: {config}")

# Initialize model
print("\nInitializing model...")
model = GPT(config)
model.init_weights()

# Count parameters (in MLX, parameters() returns a nested dict)
def count_params(tree):
    """Recursively count parameters in MLX model tree"""
    total = 0
    if isinstance(tree, dict):
        for v in tree.values():
            total += count_params(v)
    elif isinstance(tree, list):
        for v in tree:
            total += count_params(v)
    elif hasattr(tree, 'size'):  # MLX array
        total += tree.size
    return total

nparams = count_params(model.parameters())
print(f"Model has {nparams:,} parameters")

# Create dummy input
batch_size = 2
seq_len = 16
print(f"\nTesting forward pass with batch_size={batch_size}, seq_len={seq_len}...")

# Random token indices
idx = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass (training mode with loss)
print("Forward pass (training mode)...")
loss = model(idx, targets=targets)
mx.eval(loss)  # Materialize lazy computation
print(f"Loss: {loss.item():.4f}")

# Forward pass (inference mode)
print("\nForward pass (inference mode)...")
logits = model(idx)
mx.eval(logits)
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")

# Test generation
print("\nTesting generation...")
prompt_tokens = [1, 2, 3, 4, 5]
gen_tokens = []
for i, token in enumerate(model.generate(prompt_tokens, max_tokens=10, temperature=1.0)):
    gen_tokens.append(token)
    if i < 5:  # Only print first few
        print(f"Generated token {i+1}: {token}")

print(f"\n✅ All tests passed! Generated {len(gen_tokens)} tokens.")
print(f"Generated sequence: {prompt_tokens + gen_tokens}")
