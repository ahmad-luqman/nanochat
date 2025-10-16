#!/usr/bin/env python3
"""
Exercise 7.2: Model Sizes - Compare Different Architecture Scales

Learn how to:
- Create models of different sizes
- Benchmark parameter count vs. speed
- Understand size/quality tradeoffs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import time
import mlx.core as mx

print("=" * 60)
print("Model Sizes: Performance vs. Parameter Count Tradeoff")
print("=" * 60)

def benchmark_model(config_name, config):
    """Benchmark model creation and inference"""
    print(f"\n{'='*60}")
    print(f"Model: {config_name}")
    print(f"{'='*60}")

    # Create model
    model = GPT(config)
    model.init_weights()

    # Count parameters
    def count_params(tree):
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

    print(f"\nArchitecture:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding: {config.n_embd}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max sequence: {config.sequence_len}")

    print(f"\nParameters: {nparams:,} ({nparams/1e6:.2f}M)")

    # Benchmark inference
    batch_size = 1
    seq_len = 128
    dummy_input = mx.ones((batch_size, seq_len), dtype=mx.int32)

    # Warmup
    print(f"\nWarming up...", end="", flush=True)
    for _ in range(3):
        _ = model(dummy_input)
        mx.eval(model.parameters())
    print(" ✓")

    # Benchmark
    print(f"Benchmarking inference ({seq_len} tokens)...")
    num_runs = 5
    times = []

    for run in range(num_runs):
        start = time.time()
        logits = model(dummy_input)
        mx.eval(logits)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed*1000:.1f}ms")

    time_per_run = sum(times) / len(times)
    tokens_per_sec = seq_len / time_per_run

    print(f"\nInference Speed:")
    print(f"  Avg time: {time_per_run*1000:.1f}ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")

    return {
        "name": config_name,
        "params": nparams,
        "params_m": nparams/1e6,
        "time_ms": time_per_run * 1000,
        "tokens_per_sec": tokens_per_sec,
        "layers": config.n_layer,
        "embd": config.n_embd,
    }

# Different model sizes
configs = {
    "Tiny (d2)": GPTConfig(
        sequence_len=256,
        vocab_size=65536,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128
    ),
    "Small (d4)": GPTConfig(
        sequence_len=256,
        vocab_size=65536,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256
    ),
    "Medium (d6)": GPTConfig(
        sequence_len=512,
        vocab_size=65536,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=384
    ),
    "Large (d10)": GPTConfig(
        sequence_len=1024,
        vocab_size=65536,
        n_layer=10,
        n_head=8,
        n_kv_head=8,
        n_embd=512
    ),
}

results = []
for name, config in configs.items():
    result = benchmark_model(name, config)
    results.append(result)

print(f"\n{'='*60}")
print("Summary Table")
print(f"{'='*60}\n")

print(f"{'Model':<15s} | {'Parameters':>12s} | {'Time (ms)':>10s} | {'Speed (tok/s)':>12s}")
print(f"{'-'*55}")

for r in results:
    print(f"{r['name']:<15s} | {r['params']:>12,} | {r['time_ms']:>10.1f} | {r['tokens_per_sec']:>12.0f}")

print(f"\n{'='*60}")
print("Scaling Analysis")
print(f"{'='*60}\n")

# Analyze scaling
print("Parameter Growth:")
for i, r in enumerate(results):
    if i == 0:
        print(f"  {r['name']}: {r['params']:,} (baseline)")
    else:
        ratio = r['params'] / results[0]['params']
        print(f"  {r['name']}: {r['params']:,} ({ratio:.1f}x)")

print("\nSpeed Degradation:")
for i, r in enumerate(results):
    if i == 0:
        baseline_speed = r['tokens_per_sec']
        print(f"  {r['name']}: {r['tokens_per_sec']:.0f} tok/s (baseline)")
    else:
        ratio = results[0]['tokens_per_sec'] / r['tokens_per_sec']
        print(f"  {r['name']}: {r['tokens_per_sec']:.0f} tok/s ({ratio:.1f}x slower)")

print(f"\n{'='*60}")
print("Quality vs. Speed Tradeoff")
print(f"{'='*60}")

print("""
Typical Quality vs. Speed Progression:

TINY (Millions of params):
- Speed: Very fast (100+ tok/sec)
- Quality: Good for learning
- Use case: Education, prototypes
- Deployment: Any device

SMALL (Tens of millions):
- Speed: Fast (10-50 tok/sec)
- Quality: Decent for simple tasks
- Use case: Mobile, embedded
- Deployment: Edge devices, phones

MEDIUM (Hundreds of millions):
- Speed: Moderate (1-10 tok/sec)
- Quality: Good, general purpose
- Use case: Most production applications
- Deployment: Servers, laptops with GPU

LARGE (Billions+):
- Speed: Slow (0.1-1 tok/sec)
- Quality: Excellent, complex tasks
- Use case: Premium service, research
- Deployment: Datacenters, cloud

Real-world examples:
- Mobile: ~125M params (Phi-2 small)
- Standard: ~7B params (Llama-2)
- Advanced: ~13B-70B params (Llama-2 large)
- SOTA: ~70B-175B params (Llama-2 XL)
""")

print(f"\n{'='*60}")
print("Deployment Decision Tree")
print(f"{'='*60}")

print("""
Choosing model size:

1. What's your latency requirement?
   - < 50ms per token → Need small model (<100M)
   - < 500ms per token → Medium model (1-7B)
   - Can wait seconds → Large model (13-70B)

2. What's your hardware budget?
   - Mobile/edge → <100M params
   - Laptop/single GPU → <2B params
   - Server/multiple GPUs → 7-20B params
   - Datacenters → 70B+ params

3. What's your quality requirement?
   - Basic: 125M-350M
   - Good: 1-3B
   - Great: 7-13B
   - Excellent: 20-70B

4. What's the use case?
   - Information retrieval: Any size works
   - Reasoning: Needs larger model
   - Creative writing: Any size works
   - Code generation: Needs larger model
   - Translation: Needs larger model
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Create custom intermediate sizes:
   - d3, d5, d7 models
   - Plot: parameters vs. speed curve

2. Vary sequence length:
   - Same params, different seq_len
   - How does it affect speed?

3. Compare quality vs. speed:
   - Train each size on same data
   - Measure final loss
   - Plot: quality vs. speed pareto frontier

4. Estimate deployment costs:
   - What's the cost per token for each size?
   - Include GPU/CPU time
   - Find optimal size for revenue

5. Latency breakdown:
   - First token latency
   - Subsequent token latency
   - Why are they different?
""")
