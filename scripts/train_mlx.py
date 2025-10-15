#!/usr/bin/env python3
"""
Minimal training script for nanochat on MLX
Trains a GPT model using AdamW + Muon optimizers
"""
import os
import sys
import time
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW, Muon
from nanochat.data_mlx import SyntheticDataLoader
from nanochat.common_mlx import print0


def create_model(model_size='d6'):
    """Create GPT model based on size"""
    configs = {
        'd2': GPTConfig(sequence_len=256, vocab_size=2048, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
        'd6': GPTConfig(sequence_len=512, vocab_size=4096, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
        'd10': GPTConfig(sequence_len=1024, vocab_size=8192, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = GPT(config)
    model.init_weights()
    return model, config


def split_params_for_optimizer(model):
    """
    Split parameters into two groups:
    - Group 1 (AdamW): Embeddings and final layer (0D/1D params)
    - Group 2 (Muon): All 2D parameters (linear/conv layers)

    Returns:
        adamw_params: dict of parameters for AdamW
        muon_params: dict of parameters for Muon
    """
    adamw_params = {}
    muon_params = {}

    all_params = model.parameters()

    # Process embeddings
    if 'wte' in all_params:
        adamw_params['wte'] = all_params['wte']

    # Process final layer
    if 'lm_head' in all_params:
        adamw_params['lm_head'] = all_params['lm_head']

    # Process transformer blocks
    if 'h' in all_params:
        muon_params['h'] = all_params['h']

    return adamw_params, muon_params


def loss_fn(model, x, y):
    """Loss function for value_and_grad"""
    return model(x, targets=y)


def train_step(model, x, y, adamw_opt, muon_opt, use_muon=False):
    """
    Single training step using MLX's functional gradient API

    Args:
        model: GPT model
        x: Input tokens (batch_size, seq_len)
        y: Target tokens (batch_size, seq_len)
        adamw_opt: AdamW optimizer
        muon_opt: Muon optimizer (optional)
        use_muon: Whether to use Muon for 2D params

    Returns:
        loss: Scalar loss value
    """
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)

    # For now, use single optimizer for simplicity
    # TODO: Split params between AdamW and Muon in future
    if use_muon:
        muon_opt.update(model, grads)
    else:
        adamw_opt.update(model, grads)

    # Evaluate to materialize updates
    mx.eval(model.parameters())

    return loss


def estimate_mfu(model, batch_size, seq_len, dt):
    """
    Estimate model FLOPs utilization (MFU)

    Args:
        model: GPT model
        batch_size: Batch size
        seq_len: Sequence length
        dt: Time per iteration (seconds)

    Returns:
        mfu: Model FLOPs utilization as percentage
    """
    # Get FLOPs per token
    flops_per_token = model.estimate_flops()
    flops_per_iter = flops_per_token * batch_size * seq_len

    # M4 Max theoretical peak: ~14 TFLOPS for ML workloads
    # (This is a rough estimate for bfloat16 operations)
    m4_max_tflops = 14e12

    flops_achieved = flops_per_iter / dt
    mfu = 100 * flops_achieved / m4_max_tflops

    return mfu


def train(
    model_size='d6',
    batch_size=8,
    seq_len=128,
    max_steps=1000,
    learning_rate=0.01,
    weight_decay=0.0,
    use_muon=False,
    log_interval=10,
    eval_interval=100,
):
    """
    Main training loop

    Args:
        model_size: Model size (d2, d6, d10)
        batch_size: Batch size
        seq_len: Sequence length
        max_steps: Maximum training steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_muon: Whether to use Muon optimizer for 2D params
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
    """
    print0("=" * 70)
    print0("MLX Training - nanochat")
    print0("=" * 70)

    # Create model
    print0(f"\nInitializing {model_size} model...")
    model, config = create_model(model_size)

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
    print0(f"Model parameters: {nparams:,} ({nparams/1e6:.2f}M)")
    print0(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")

    # Create optimizers
    print0(f"\nSetting up optimizers...")
    print0(f"  Learning rate: {learning_rate}")
    print0(f"  Weight decay: {weight_decay}")

    # Scale learning rate by model dimension (like original)
    model_dim = config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    scaled_lr = learning_rate * dmodel_lr_scale
    print0(f"  Scaled LR (∝1/√(d/768)): {scaled_lr:.6f}")

    adamw = AdamW(
        learning_rate=scaled_lr,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=weight_decay
    )

    muon = None
    if use_muon:
        muon = Muon(
            learning_rate=0.02 * dmodel_lr_scale,
            momentum=0.95,
            nesterov=True,
            ns_steps=5
        )
        print0(f"  Using Muon for 2D parameters")
    else:
        print0(f"  Using AdamW for all parameters")

    # Create data loader
    print0(f"\nSetting up data loader...")
    print0(f"  Batch size: {batch_size}")
    print0(f"  Sequence length: {seq_len}")
    print0(f"  Vocab size: {config.vocab_size}")

    num_batches_per_epoch = 100
    dataloader = SyntheticDataLoader(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        num_batches=num_batches_per_epoch
    )

    # Training loop
    print0("\n" + "=" * 70)
    print0("Starting training...")
    print0("=" * 70)

    step = 0
    epoch = 0
    running_loss = 0.0
    start_time = time.time()

    while step < max_steps:
        epoch += 1

        for x, y in dataloader:
            step += 1

            # Training step
            step_start = time.time()
            loss = train_step(model, x, y, adamw, muon, use_muon=use_muon)
            mx.eval(loss)
            step_time = time.time() - step_start

            running_loss += loss.item()

            # Logging
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                tokens_per_sec = (batch_size * seq_len) / step_time
                mfu = estimate_mfu(model, batch_size, seq_len, step_time)
                elapsed = time.time() - start_time

                print0(f"Step {step:4d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {step_time*1000:.1f}ms | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"MFU: {mfu:.2f}% | "
                      f"Elapsed: {elapsed:.1f}s")

                running_loss = 0.0

            # Stop if max steps reached
            if step >= max_steps:
                break

    # Final summary
    total_time = time.time() - start_time
    print0("\n" + "=" * 70)
    print0("Training complete!")
    print0("=" * 70)
    print0(f"Total steps: {step}")
    print0(f"Total time: {total_time:.1f}s")
    print0(f"Avg time per step: {total_time/step*1000:.1f}ms")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train nanochat model on MLX")
    parser.add_argument("--model-size", type=str, default="d6", choices=['d2', 'd6', 'd10'],
                       help="Model size (d2, d6, d10)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--use-muon", action="store_true",
                       help="Use Muon optimizer for 2D parameters")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Steps between logging")

    args = parser.parse_args()

    # Train model
    model = train(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_muon=args.use_muon,
        log_interval=args.log_interval,
    )
