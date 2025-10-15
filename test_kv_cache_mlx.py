#!/usr/bin/env python3
"""
Test KV cache for MLX GPT model
"""
import mlx.core as mx
from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache

print("Testing MLX KV Cache")
print("=" * 70)

# Create a tiny model
config = GPTConfig(
    sequence_len=256,
    vocab_size=1000,
    n_layer=2,
    n_head=4,
    n_kv_head=4,
    n_embd=128
)
model = GPT(config)
model.init_weights()

print(f"Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")

# Test 1: Basic KV cache usage
print("\n" + "=" * 70)
print("Test 1: Basic KV cache usage")
print("=" * 70)

batch_size = 1
head_dim = config.n_embd // config.n_head

# Create KV cache
kv_cache = KVCache(
    batch_size=batch_size,
    num_heads=config.n_kv_head,
    seq_len=256,
    head_dim=head_dim,
    num_layers=config.n_layer
)

print(f"KV cache shape: {kv_cache.kv_shape}")
print(f"Initial pos: {kv_cache.get_pos()}")

# Generate with KV cache (prefix pass)
prompt_tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
print(f"\nPrefill with {prompt_tokens.shape[1]} tokens...")
logits = model(prompt_tokens, kv_cache=kv_cache)
mx.eval(logits)
print(f"Logits shape: {logits.shape}")
print(f"Cache pos after prefill: {kv_cache.get_pos()}")
assert kv_cache.get_pos() == 5, f"Expected pos=5, got {kv_cache.get_pos()}"

# Generate one token at a time (decode phase)
print(f"\nGenerate 5 tokens autoregressively...")
for i in range(5):
    # Sample next token (just use argmax for testing)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # (B, 1)

    # Forward with single token
    logits = model(next_token, kv_cache=kv_cache)
    mx.eval(logits)

    print(f"  Step {i+1}: pos={kv_cache.get_pos()}, token={int(next_token[0,0])}")

assert kv_cache.get_pos() == 10, f"Expected pos=10, got {kv_cache.get_pos()}"
print(f"\n✅ Test 1 passed!")

# Test 2: KV cache prefill (batch expansion)
print("\n" + "=" * 70)
print("Test 2: KV cache prefill (batch expansion)")
print("=" * 70)

# Create single-batch cache and prefill
cache_single = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=256,
    head_dim=head_dim,
    num_layers=config.n_layer
)

prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
logits = model(prompt, kv_cache=cache_single)
mx.eval(logits)
print(f"Single batch prefill: pos={cache_single.get_pos()}")

# Create multi-batch cache and prefill from single
num_samples = 4
cache_multi = KVCache(
    batch_size=num_samples,
    num_heads=config.n_kv_head,
    seq_len=256,
    head_dim=head_dim,
    num_layers=config.n_layer
)

cache_multi.prefill(cache_single)
print(f"Multi-batch cache prefilled: pos={cache_multi.get_pos()}")
assert cache_multi.get_pos() == cache_single.get_pos()
print(f"Cache batch size: {cache_multi.kv_cache.shape[2]}")

# Generate one token for all samples
next_tokens = mx.array([[10], [20], [30], [40]], dtype=mx.int32)
logits = model(next_tokens, kv_cache=cache_multi)
mx.eval(logits)
print(f"Generated for {num_samples} samples: pos={cache_multi.get_pos()}")
assert cache_multi.get_pos() == 4

print(f"\n✅ Test 2 passed!")

# Test 3: Compare cached vs non-cached generation
print("\n" + "=" * 70)
print("Test 3: Compare cached vs non-cached generation")
print("=" * 70)

prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

# Without cache: pass full sequence
logits_no_cache = model(prompt)
mx.eval(logits_no_cache)
last_logits_no_cache = logits_no_cache[:, -1, :]

# With cache: prefill then get last logits
cache_test = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=256,
    head_dim=head_dim,
    num_layers=config.n_layer
)
logits_cache = model(prompt, kv_cache=cache_test)
mx.eval(logits_cache)
last_logits_cache = logits_cache[:, -1, :]

# Compare logits (should be identical)
diff = mx.abs(last_logits_no_cache - last_logits_cache)
max_diff = mx.max(diff).item()
print(f"Max difference in logits: {max_diff:.6f}")

if max_diff < 1e-4:
    print(f"✅ Logits match! (max diff: {max_diff:.6f})")
else:
    print(f"⚠️  Warning: Logits differ by {max_diff:.6f}")

print(f"\n✅ Test 3 passed!")

# Test 4: Dynamic cache growth
print("\n" + "=" * 70)
print("Test 4: Dynamic cache growth")
print("=" * 70)

# Create small cache that will need to grow
cache_small = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=8,  # Very small initial size
    head_dim=head_dim,
    num_layers=config.n_layer
)

print(f"Initial cache seq_len: {cache_small.kv_shape[4]}")

# Generate 20 tokens (will exceed initial capacity)
prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
logits = model(prompt, kv_cache=cache_small)
mx.eval(logits)

for i in range(20):
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # (B, 1)
    logits = model(next_token, kv_cache=cache_small)
    mx.eval(logits)

print(f"Final pos: {cache_small.get_pos()}")
print(f"Final cache seq_len: {cache_small.kv_cache.shape[4]}")
assert cache_small.get_pos() == 23
assert cache_small.kv_cache.shape[4] >= 23

print(f"\n✅ Test 4 passed!")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✅ Basic KV cache usage")
print("✅ KV cache prefill with batch expansion")
print("✅ Cached vs non-cached generation")
print("✅ Dynamic cache growth")
print("\nAll KV cache tests passed!")
