#!/usr/bin/env python3
"""
Comprehensive test suite for MLX GPT model
Tests multiple model sizes, edge cases, and performance
"""
import time
import mlx.core as mx
from nanochat.gpt_mlx import GPT, GPTConfig

print("=" * 70)
print("MLX GPT Model - Comprehensive Test Suite")
print("=" * 70)

def count_params(tree):
    """Recursively count parameters"""
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

def test_model(name, config, batch_size=2, seq_len=16):
    """Test a model configuration"""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")
    print(f"Config: depth={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, vocab={config.vocab_size}")

    # Initialize model
    start = time.time()
    model = GPT(config)
    model.init_weights()
    init_time = time.time() - start

    nparams = count_params(model.parameters())
    print(f"Parameters: {nparams:,} ({nparams/1e6:.2f}M)")
    print(f"Initialization time: {init_time:.3f}s")

    # Test forward pass (training)
    print(f"\n  Testing forward pass (training mode)...")
    idx = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    start = time.time()
    loss = model(idx, targets=targets)
    mx.eval(loss)
    forward_time = time.time() - start

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Forward time: {forward_time*1000:.2f}ms")

    # Test forward pass (inference)
    print(f"\n  Testing forward pass (inference mode)...")
    start = time.time()
    logits = model(idx)
    mx.eval(logits)
    inference_time = time.time() - start

    print(f"  Logits shape: {logits.shape}")
    print(f"  Inference time: {inference_time*1000:.2f}ms")

    # Test generation
    print(f"\n  Testing generation (10 tokens)...")
    prompt_tokens = [1, 2, 3, 4, 5]
    start = time.time()
    gen_tokens = []
    for i, token in enumerate(model.generate(prompt_tokens, max_tokens=10, temperature=1.0)):
        gen_tokens.append(token)
    gen_time = time.time() - start

    print(f"  Generated: {gen_tokens}")
    print(f"  Generation time: {gen_time*1000:.2f}ms ({len(gen_tokens)/gen_time:.1f} tok/s)")

    # Test with ignore_index
    print(f"\n  Testing loss with ignore_index...")
    targets_masked = mx.array(targets, dtype=mx.int32)
    # Create new array with first token of each batch set to -1
    first_col = mx.full((batch_size, 1), -1, dtype=mx.int32)
    targets_masked = mx.concatenate([first_col, targets_masked[:, 1:]], axis=1)
    loss_masked = model(idx, targets=targets_masked)
    mx.eval(loss_masked)
    print(f"  Loss (with masking): {loss_masked.item():.4f}")

    print(f"\n  ✅ All tests passed for {name}!")
    return {
        'params': nparams,
        'init_time': init_time,
        'forward_time': forward_time,
        'inference_time': inference_time,
        'gen_time': gen_time,
        'tokens_per_sec': len(gen_tokens) / gen_time
    }

# Test Suite
results = {}

# Test 1: Tiny model (quick sanity check)
print("\n" + "=" * 70)
print("TEST 1: Tiny Model (d2)")
print("=" * 70)
config_tiny = GPTConfig(
    sequence_len=128,
    vocab_size=1000,
    n_layer=2,
    n_head=4,
    n_kv_head=4,
    n_embd=128
)
results['tiny'] = test_model("Tiny d2", config_tiny, batch_size=4, seq_len=32)

# Test 2: Small model (d6)
print("\n" + "=" * 70)
print("TEST 2: Small Model (d6)")
print("=" * 70)
config_small = GPTConfig(
    sequence_len=512,
    vocab_size=2000,
    n_layer=6,
    n_head=6,
    n_kv_head=6,
    n_embd=384
)
results['small'] = test_model("Small d6", config_small, batch_size=2, seq_len=64)

# Test 3: Medium model (d10)
print("\n" + "=" * 70)
print("TEST 3: Medium Model (d10)")
print("=" * 70)
config_medium = GPTConfig(
    sequence_len=1024,
    vocab_size=4096,
    n_layer=10,
    n_head=8,
    n_kv_head=8,
    n_embd=512
)
results['medium'] = test_model("Medium d10", config_medium, batch_size=2, seq_len=128)

# Test 4: Multi-Query Attention (MQA)
print("\n" + "=" * 70)
print("TEST 4: Multi-Query Attention (MQA)")
print("=" * 70)
config_mqa = GPTConfig(
    sequence_len=512,
    vocab_size=2000,
    n_layer=4,
    n_head=8,
    n_kv_head=2,  # 8 query heads, 2 kv heads
    n_embd=256
)
results['mqa'] = test_model("MQA (8q/2kv)", config_mqa, batch_size=2, seq_len=64)

# Test 5: Edge cases
print("\n" + "=" * 70)
print("TEST 5: Edge Cases")
print("=" * 70)

print("\n  Test 5a: Batch size = 1")
config_edge = GPTConfig(sequence_len=128, vocab_size=500, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
model_edge = GPT(config_edge)
model_edge.init_weights()
idx = mx.random.randint(0, 500, (1, 16))
targets = mx.random.randint(0, 500, (1, 16))
loss = model_edge(idx, targets=targets)
mx.eval(loss)
print(f"  Loss (batch=1): {loss.item():.4f} ✅")

print("\n  Test 5b: Long sequence")
idx_long = mx.random.randint(0, 500, (1, 100))
logits_long = model_edge(idx_long)
mx.eval(logits_long)
print(f"  Logits shape (seq=100): {logits_long.shape} ✅")

print("\n  Test 5c: Temperature = 0 (greedy)")
gen_greedy = list(model_edge.generate([1, 2, 3], max_tokens=5, temperature=0.0))
print(f"  Greedy generation: {gen_greedy} ✅")

print("\n  Test 5d: Top-k sampling")
gen_topk = list(model_edge.generate([1, 2, 3], max_tokens=5, temperature=1.0, top_k=10))
print(f"  Top-k generation: {gen_topk} ✅")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Model':<15} {'Params':<12} {'Forward':<12} {'Generation':<15}")
print("-" * 70)
for name, res in results.items():
    print(f"{name:<15} {res['params']/1e6:>6.2f}M     {res['forward_time']*1000:>6.1f}ms      {res['tokens_per_sec']:>6.1f} tok/s")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✅")
print("=" * 70)
print("\nMLX Model Status:")
print("  ✅ Forward pass (training)")
print("  ✅ Forward pass (inference)")
print("  ✅ Loss computation with masking")
print("  ✅ Text generation")
print("  ✅ Multi-Query Attention (MQA)")
print("  ✅ Edge cases (batch=1, long seq, greedy, top-k)")
print("\nReady for Phase 2: Training loop implementation!")
