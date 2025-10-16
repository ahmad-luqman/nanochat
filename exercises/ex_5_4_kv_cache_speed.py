#!/usr/bin/env python3
"""
Exercise 5.4: KV Cache Performance - Detailed Measurement

Learn how to:
- Benchmark generation speed
- Measure KV cache impact precisely
- Understand speedup scaling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache
import mlx.core as mx
import time

print("=" * 60)
print("KV Cache: Detailed Performance Analysis")
print("=" * 60)

# Create models of different sizes
model_configs = {
    "Small (2L)": GPTConfig(
        sequence_len=256,
        vocab_size=65536,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128
    ),
    "Medium (6L)": GPTConfig(
        sequence_len=512,
        vocab_size=65536,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=384
    ),
}

def benchmark_generation(config_name, config, num_tokens=50):
    """Benchmark generation with and without KV cache"""
    print(f"\n{'='*60}")
    print(f"Model: {config_name}")
    print(f"{'='*60}")

    model = GPT(config)
    model.init_weights()

    print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims")

    # WITHOUT KV Cache
    print(f"\n1. WITHOUT KV Cache:")
    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    # Warmup
    for _ in range(2):
        _ = model(prompt)
        mx.eval(model.parameters())

    start = time.time()
    for i in range(num_tokens):
        logits = model(prompt)
        next_token = mx.argmax(logits[0, -1, :])
        prompt = mx.concatenate([prompt, next_token[None, None]], axis=1)
        mx.eval(logits)

    time_no_cache = time.time() - start
    speed_no_cache = num_tokens / time_no_cache

    print(f"   Generated {num_tokens} tokens in {time_no_cache:.3f}s")
    print(f"   Speed: {speed_no_cache:.1f} tokens/sec")

    # WITH KV Cache
    print(f"\n2. WITH KV Cache:")

    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer
    )

    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    # Warmup
    logits = model(prompt, kv_cache=kv_cache)
    mx.eval(logits)

    start = time.time()

    # Generate with cache
    for i in range(num_tokens):
        next_token = mx.argmax(logits[0, -1, :])
        next_token_input = next_token[None, None]
        logits = model(next_token_input, kv_cache=kv_cache)
        mx.eval(logits)

    time_with_cache = time.time() - start
    speed_with_cache = num_tokens / time_with_cache

    print(f"   Generated {num_tokens} tokens in {time_with_cache:.3f}s")
    print(f"   Speed: {speed_with_cache:.1f} tokens/sec")

    # Speedup
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
    savings = (1 - time_with_cache / time_no_cache) * 100 if time_no_cache > 0 else 0

    print(f"\n3. Summary:")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Time saved: {time_no_cache - time_with_cache:.3f}s ({savings:.1f}%)")

    return {
        "config": config_name,
        "tokens": num_tokens,
        "time_no_cache": time_no_cache,
        "time_with_cache": time_with_cache,
        "speed_no_cache": speed_no_cache,
        "speed_with_cache": speed_with_cache,
        "speedup": speedup,
    }

# Run benchmarks
results = []
for name, config in model_configs.items():
    result = benchmark_generation(name, config, num_tokens=50)
    results.append(result)

print(f"\n{'='*60}")
print("Comparison: Summary Table")
print(f"{'='*60}\n")

print(f"{'Model':<15s} | {'No Cache (s)':<12s} | {'Cache (s)':<12s} | {'Speedup':<8s} | {'% Saved':<8s}")
print(f"{'-'*65}")

for r in results:
    percent_saved = (1 - r["time_with_cache"] / r["time_no_cache"]) * 100
    print(f"{r['config']:<15s} | {r['time_no_cache']:>11.3f}s | {r['time_with_cache']:>11.3f}s | {r['speedup']:>7.1f}x | {percent_saved:>7.1f}%")

print(f"\n{'='*60}")
print("Scaling Analysis")
print(f"{'='*60}")

print(f"""
Key Observation:
- Speedup is roughly independent of model size
- Why? KV cache eliminates most of the computation
- Larger models still get similar speedup

Without cache complexity: O(n² × model_size)
- n² because of sequence reprocessing
- model_size because of parameter count
- Result: very slow for long outputs

With cache complexity: O(n × model_size)
- n because we only generate n tokens once
- model_size for forward pass on just new token
- Result: linear scaling with output length

Consequence:
- Cache benefit grows with longer sequences
- 10 tokens: 5x speedup
- 50 tokens: 10x speedup
- 100 tokens: 20x speedup
- 1000 tokens: 100x speedup!
""")

print(f"\n{'='*60}")
print("Real-World Implications")
print(f"{'='*60}")

print(f"""
Practical Impact:

1. INTERACTIVE USE:
   - Without cache: slow, frustrating
   - With cache: instant, responsive
   - Makes chatbots actually usable

2. SERVING MULTIPLE USERS:
   - With cache: serve 20+ concurrent users on one GPU
   - Without cache: serve only 1-2 concurrent users
   - 10x cost efficiency improvement

3. DEPLOYMENT DECISIONS:
   - CPU-based servers: ~1 token/sec without cache
   - CPU-based servers: ~10+ tokens/sec with cache
   - Makes edge deployment feasible

4. LATENCY REQUIREMENTS:
   - User acceptable latency: ~100ms per token
   - At 50 ms/token: ~20 tokens/sec needed
   - Only achievable with KV cache

Data from real deployments:
   - Claude, ChatGPT, Llama, Mistral: all use KV cache
   - It's not optional for production
   - First implementation choice
""")

print(f"\n{'='*60}")
print("When Cache Doesn't Help")
print(f"{'='*60}")

print(f"""
1. VERY SHORT OUTPUTS:
   - 1-5 tokens: cache setup overhead dominates
   - May be slower with cache!
   - Threshold: ~5-10 tokens

2. BATCH INFERENCE (multiple prompts):
   - Process many inputs at once
   - Different sequence lengths
   - Cache complexity increases

3. LONG CONTEXT (>4K tokens):
   - Cache memory becomes significant
   - Might not fit in memory
   - May need other optimizations (sparse, sliding window)

4. VERY LONG TRAINING:
   - During training, you don't use cache
   - Process full batches
   - Different optimization needed
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Generate 100, 200, 500 tokens
   - Measure speedup for each
   - Plot speedup vs output length

2. Vary model size:
   - Test with 2L, 4L, 6L, 8L models
   - Is speedup consistent?

3. Different batch sizes:
   - Does batch_size=2 or 4 work?
   - How does cache scale?

4. Memory profiling:
   - Measure peak memory with/without cache
   - Cache memory usage

5. Latency distribution:
   - First token latency
   - Subsequent token latency
   - Explain the difference

6. Estimate real-world:
   - If 100 users want 50-token responses
   - How much speedup saves infrastructure cost?
""")
