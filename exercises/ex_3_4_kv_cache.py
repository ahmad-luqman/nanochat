#!/usr/bin/env python3
"""
Exercise 3.4: KV Cache - Speed Up Generation

Learn how to:
- Understand why KV cache helps
- Use KV cache in generation
- Measure speedup
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache
import mlx.core as mx
import time

print("=" * 60)
print("KV Cache: Fast Generation Optimization")
print("=" * 60)

# Medium-sized model
config = GPTConfig(
    sequence_len=512,
    vocab_size=65536,
    n_layer=4,
    n_head=6,
    n_kv_head=6,
    n_embd=384
)
model = GPT(config)
model.init_weights()

print(f"\nModel configuration:")
print(f"  Layers: {config.n_layer}")
print(f"  Heads: {config.n_head}")
print(f"  Embedding: {config.n_embd}")

print(f"\n{'='*60}")
print("Understanding the Problem Without KV Cache")
print(f"{'='*60}")

print("""
During generation, we process tokens one at a time:

Step 1: Input [A]         → Process all layers → Output token
Step 2: Input [A, B]      → Process all layers → Output token
Step 3: Input [A, B, C]   → Process all layers → Output token
Step 4: Input [A, B, C, D] → Process all layers → Output token

Problem: We recompute [A], [A,B], [A,B,C] every time!
- A is processed 4 times
- B is processed 3 times
- C is processed 2 times
- D is processed 1 time

Result: O(n²) computation for n tokens!
        Quadratic slowdown as we generate more tokens.
""")

print(f"\n{'='*60}")
print("How KV Cache Solves It")
print(f"{'='*60}")

print("""
KV Cache = Key/Value cache from attention

In attention: output = softmax(Q @ K^T) @ V
Problem: We recompute K and V for previous tokens
Solution: Cache K and V values!

With KV cache:

Step 1: Input [A]         → Compute Q,K,V for A → Cache K,V(A) → Output
Step 2: Input [B]         → Compute Q for B, use cached K,V(A) → Output
Step 3: Input [C]         → Compute Q for C, use cached K,V(A),K,V(B) → Output
Step 4: Input [D]         → Compute Q for D, use cached K,V(A-C) → Output

Result: Each layer only processes the new token!
        O(n) computation instead of O(n²).

Speedup: ~10x for 100 tokens, ~100x for 1000 tokens!
""")

print(f"\n{'='*60}")
print("KV Cache in Action")
print(f"{'='*60}")

# Generate WITHOUT KV cache (slow - recomputes everything)
print("\n1. WITHOUT KV Cache (slow):")
prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

start = time.time()
for i in range(20):
    # Reprocess entire sequence each time!
    logits = model(prompt)
    next_token = mx.argmax(logits[0, -1, :])
    prompt = mx.concatenate([prompt, next_token[None, None]], axis=1)
    mx.eval(logits)

no_cache_time = time.time() - start
print(f"   Generated 20 tokens in {no_cache_time:.3f}s")
print(f"   Speed: {20/no_cache_time:.1f} tokens/sec")

# Generate WITH KV cache (fast - only process new token)
print("\n2. WITH KV Cache (fast):")

kv_cache = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=config.sequence_len,
    head_dim=config.n_embd // config.n_head,
    num_layers=config.n_layer
)

prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

start = time.time()

# Process prompt once
logits = model(prompt, kv_cache=kv_cache)
mx.eval(logits)

# Generate tokens one at a time
for i in range(20):
    next_token = mx.argmax(logits[0, -1, :])
    next_token_input = next_token[None, None]
    logits = model(next_token_input, kv_cache=kv_cache)
    mx.eval(logits)

cache_time = time.time() - start
print(f"   Generated 20 tokens in {cache_time:.3f}s")
print(f"   Speed: {20/cache_time:.1f} tokens/sec")

print(f"\n{'='*60}")
print("Speedup Calculation")
print(f"{'='*60}")

speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
print(f"\nSpeedup: {speedup:.1f}x faster with KV cache!")
print(f"  Without cache: {no_cache_time:.3f}s")
print(f"  With cache:    {cache_time:.3f}s")
print(f"  Saved time:    {no_cache_time - cache_time:.3f}s")

print(f"\n{'='*60}")
print("Memory vs Speed Tradeoff")
print(f"{'='*60}")

# Calculate cache size
cache_size_mb = (
    2 *  # Two (K and V)
    config.n_layer *  # Per layer
    config.sequence_len *  # Sequence length
    config.n_head *  # Heads
    (config.n_embd // config.n_head) *  # Head dimension
    2 / (1024 * 1024)  # Convert to MB (2 bytes per float)
)

print(f"\nKV Cache Memory Usage:")
print(f"  Layers: {config.n_layer}")
print(f"  Sequence length: {config.sequence_len}")
print(f"  Cache size: ~{cache_size_mb:.1f} MB")
print(f"\nTradeoff:")
print(f"  - Faster generation: 10-100x speedup")
print(f"  - More memory: a few MB per model")
print(f"  - Worth it: absolutely yes!")

print(f"\n{'='*60}")
print("Real-World Impact")
print(f"{'='*60}")

print("""
Generation times for 100 tokens:

Without KV cache:
- 1B param model: ~10-20 seconds
- 7B param model: ~50-100 seconds
- 13B param model: ~100-200 seconds

With KV cache:
- 1B param model: ~0.5-1 second (20x faster!)
- 7B param model: ~2-3 seconds (20x faster!)
- 13B param model: ~4-6 seconds (20x faster!)

Difference:
- Without cache: too slow for interactive use
- With cache: real-time conversation possible

For deployment:
- Every production LLM uses KV cache
- Essential for serving multiple users
- Massive impact on cost efficiency
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Generate 50 tokens and measure speedup
2. Generate 100 tokens - does speedup improve?
3. Vary sequence length (256, 512, 1024)
4. Vary number of layers (2, 4, 6, 8)
5. Calculate speedup curve (tokens vs speedup)
6. Estimate cost of generation without vs with cache
7. Why is the speedup factor consistent?
8. What happens if cache runs out of memory?
""")
