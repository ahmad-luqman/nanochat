#!/usr/bin/env python3
"""
Exercise 6.4: Checkpoint Analysis - Inspect Model Metadata

Learn how to:
- Load checkpoint metadata
- Analyze model configurations
- Estimate model size
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os

print("=" * 60)
print("Checkpoint Analysis: Understanding Model Metadata")
print("=" * 60)

# Checkpoints to analyze
checkpoint_dirs = [
    "checkpoints/d10_pretrain_causal_fix",
    "checkpoints/d10_sft",
]

def analyze_checkpoint(ckpt_dir):
    """Analyze a checkpoint directory"""
    if not os.path.exists(ckpt_dir):
        print(f"\n⚠️  Checkpoint not found: {ckpt_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    # Find best checkpoint
    best_path = os.path.join(ckpt_dir, "best.npz.meta.json")
    if not os.path.exists(best_path):
        print(f"  ⚠️  No metadata found at: {best_path}")
        return None

    with open(best_path, 'r') as f:
        meta = json.load(f)

    print(f"\nBasic Information:")
    print(f"  Step: {meta.get('step', 'N/A')}")
    print(f"  Loss: {meta.get('loss', 'N/A'):.4f}" if isinstance(meta.get('loss'), (int, float)) else f"  Loss: {meta.get('loss', 'N/A')}")
    print(f"  Timestamp: {meta.get('timestamp', 'N/A')}")

    if 'model_config' in meta:
        config = meta['model_config']
        print(f"\nModel Configuration:")
        print(f"  Layers (n_layer): {config.get('n_layer')}")
        print(f"  Attention Heads (n_head): {config.get('n_head')}")
        print(f"  KV Heads (n_kv_head): {config.get('n_kv_head')}")
        print(f"  Embedding dimension (n_embd): {config.get('n_embd')}")
        print(f"  Vocabulary size: {config.get('vocab_size')}")
        print(f"  Max sequence length: {config.get('sequence_len')}")

        # Calculate model size
        n_layer = config.get('n_layer', 0)
        n_embd = config.get('n_embd', 0)
        n_head = config.get('n_head', 0)
        vocab_size = config.get('vocab_size', 0)
        seq_len = config.get('sequence_len', 0)

        if all([n_layer, n_embd, vocab_size]):
            # Parameter estimation
            # Token embedding: vocab_size * n_embd
            token_embed = vocab_size * n_embd

            # Position embedding: sequence_len * n_embd
            pos_embed = seq_len * n_embd

            # Per transformer block:
            # - LayerNorm: 2 * n_embd (scale + bias)
            # - Attention Q, K, V projections: 3 * n_embd^2
            # - Attention output projection: n_embd^2
            # - MLP: n_embd * (4*n_embd) + (4*n_embd) * n_embd
            # Total per block ≈ 12 * n_embd^2 + some layer norms

            transformer_params = (
                n_layer * (
                    2 * n_embd +  # layer norms
                    3 * n_embd * n_embd +  # QKV
                    n_embd * n_embd +  # out projection
                    2 * 4 * n_embd * n_embd  # MLP (2 linear layers)
                )
            )

            # Output projection: n_embd * vocab_size
            output_proj = n_embd * vocab_size

            # Total parameters
            total_params = token_embed + pos_embed + transformer_params + output_proj

            # Size in MB (assuming float32: 4 bytes per param, or bfloat16: 2 bytes)
            size_float32_mb = total_params * 4 / (1024 * 1024)
            size_bf16_mb = total_params * 2 / (1024 * 1024)

            print(f"\nParameter Estimation:")
            print(f"  Token embedding: {token_embed:,}")
            print(f"  Position embedding: {pos_embed:,}")
            print(f"  Transformer blocks: {transformer_params:,}")
            print(f"  Output projection: {output_proj:,}")
            print(f"  {'─' * 50}")
            print(f"  Total: {total_params:,} params ({total_params/1e6:.2f}M)")

            print(f"\nModel Size (rough estimate):")
            print(f"  Float32 (4 bytes): {size_float32_mb:.1f} MB")
            print(f"  BFloat16 (2 bytes): {size_bf16_mb:.1f} MB (typical for deployment)")

            print(f"\nContext:")
            if total_params < 1e6:
                context = "Tiny model (for learning/testing)"
            elif total_params < 10e6:
                context = "Small model (edge devices)"
            elif total_params < 100e6:
                context = "Medium model (reasonable inference)"
            elif total_params < 1e9:
                context = "Large model (GPU-class)"
            else:
                context = "Very large model (datacenter-class)"
            print(f"  Model type: {context}")

    if 'optimizer_type' in meta:
        print(f"\nOptimizer:")
        print(f"  Type: {meta['optimizer_type']}")

    if 'metadata' in meta:
        print(f"\nCustom Metadata:")
        for key, value in meta['metadata'].items():
            print(f"  {key}: {value}")

    # Check checkpoint files
    print(f"\nCheckpoint Files:")
    best_npz = os.path.join(ckpt_dir, "best.npz")
    if os.path.exists(best_npz):
        size_mb = os.path.getsize(best_npz) / (1024 * 1024)
        print(f"  best.npz: {size_mb:.1f} MB (model weights)")

    best_opt = os.path.join(ckpt_dir, "best.npz.opt.npz")
    if os.path.exists(best_opt):
        size_mb = os.path.getsize(best_opt) / (1024 * 1024)
        print(f"  best.npz.opt.npz: {size_mb:.1f} MB (optimizer state)")

    return meta

# Analyze all checkpoints
all_meta = {}
for ckpt_dir in checkpoint_dirs:
    meta = analyze_checkpoint(ckpt_dir)
    if meta:
        all_meta[ckpt_dir] = meta

print(f"\n{'='*60}")
print("Checkpoint Comparison")
print(f"{'='*60}")

if len(all_meta) > 1:
    print(f"\n{'Checkpoint':<40s} | {'Stage':<10s} | {'Loss':<10s} | {'Size (M)':<10s}")
    print(f"{'-'*75}")

    for ckpt_dir, meta in all_meta.items():
        name = os.path.basename(ckpt_dir)
        config = meta.get('model_config', {})
        loss = meta.get('loss', 0)
        step = meta.get('step', 0)

        # Estimate size
        n_embd = config.get('n_embd', 0)
        n_layer = config.get('n_layer', 0)
        vocab = config.get('vocab_size', 0)

        params = (
            n_layer * 12 * n_embd * n_embd +
            vocab * n_embd +
            config.get('sequence_len', 256) * n_embd
        )
        params_m = params / 1e6

        stage = "Pretrain" if "pretrain" in ckpt_dir else "SFT"
        print(f"{name:<40s} | {stage:<10s} | {loss:<10.4f} | {params_m:<10.1f}")

print(f"\n{'='*60}")
print("Key Insights")
print(f"{'='*60}")

print("""
Checkpoint Analysis tells you:

1. TRAINING PROGRESS:
   - Step: How long training ran
   - Loss: Model performance (lower is better)
   - Improvement: Can see loss trend

2. MODEL ARCHITECTURE:
   - Layers/heads/dims: Exactly how big
   - Parameter count: Computational cost
   - Sequence length: Context window

3. STORAGE REQUIREMENTS:
   - Model weights: Small (2-4 bytes per param)
   - Optimizer state: 2-10x model size during training
   - Important for storage planning

4. DEPLOYMENT CONSIDERATIONS:
   - Model size: RAM needed for serving
   - Inference speed: Proportional to parameters
   - Memory budget: Float32 vs BFloat16 tradeoff

5. VERSION CONTROL:
   - Can track different model checkpoints
   - Understand what improved when
   - Reproduce results from specific step
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Load checkpoints at different training steps
   - Compare loss progression
   - See how loss improves over time

2. Compare model architectures:
   - d2 vs d6 vs d10
   - How do parameters scale?
   - How does loss scale?

3. Estimate inference costs:
   - Tokens per second per size
   - Memory required per model
   - Cost per 1M tokens

4. Track training efficiency:
   - Loss reduction per step
   - Loss reduction per training hour
   - Find "diminishing returns" point

5. Reproduce results:
   - Load checkpoint
   - Run on same test data
   - Verify reproducibility
""")
