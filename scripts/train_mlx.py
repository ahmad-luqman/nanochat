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
from nanochat.data_mlx import SyntheticDataLoader, TextDataLoader
from nanochat.common_mlx import print0
from nanochat.checkpoint_mlx import save_checkpoint, load_checkpoint, get_latest_checkpoint
from nanochat.lr_scheduler_mlx import CosineAnnealingLR
from nanochat.metrics_mlx import validate, compute_bpb, load_token_bytes, MetricsLogger, print_metrics_summary


def create_model(model_size='d6'):
    """Create GPT model based on size"""
    configs = {
        'd2': GPTConfig(sequence_len=256, vocab_size=2048, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
        'd6': GPTConfig(sequence_len=512, vocab_size=4096, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
        'd10': GPTConfig(sequence_len=1024, vocab_size=8192, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
        'd14': GPTConfig(sequence_len=1024, vocab_size=65536, n_layer=14, n_head=12, n_kv_head=12, n_embd=768),
        'd20': GPTConfig(sequence_len=1024, vocab_size=65536, n_layer=20, n_head=16, n_kv_head=16, n_embd=1024),
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
    eval_interval=500,
    val_batches=100,
    use_real_data=False,
    checkpoint_dir=None,
    save_interval=1000,
    resume_from=None,
    warmup_steps=0,
    min_lr_ratio=0.1,
):
    """
    Main training loop

    Args:
        model_size: Model size (d2, d6, d10, d14, d20)
        batch_size: Batch size
        seq_len: Sequence length
        max_steps: Maximum training steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_muon: Whether to use Muon optimizer for 2D params
        log_interval: Steps between logging
        eval_interval: Steps between validation (default: 500)
        val_batches: Number of validation batches to run (default: 100)
        use_real_data: Use real data from parquet files instead of synthetic
        checkpoint_dir: Directory to save checkpoints (None = no saving)
        save_interval: Steps between checkpoint saves
        resume_from: Path to checkpoint to resume from (None = fresh start)
        warmup_steps: Number of warmup steps for LR schedule (0 = no warmup)
        min_lr_ratio: Minimum LR as fraction of max LR (for cosine annealing)
    """
    print0("=" * 70)
    print0("MLX Training - nanochat")
    print0("=" * 70)

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        print0(f"\nResuming from checkpoint: {resume_from}")
        # Load config first to create model
        import json
        meta_path = resume_from + ".meta.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Create model with config from checkpoint
        config_dict = metadata['model_config']
        config = GPTConfig(**config_dict)
        model = GPT(config)
        start_step = metadata['step']
        print0(f"  Resuming from step {start_step}")
    else:
        # Create model from scratch
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

    # Create LR scheduler
    scheduler = None
    if warmup_steps > 0:
        print0(f"\nSetting up LR scheduler...")
        print0(f"  Schedule: Cosine annealing with warmup")
        print0(f"  Warmup steps: {warmup_steps}")
        print0(f"  Max LR: {scaled_lr:.6f}")
        print0(f"  Min LR: {scaled_lr * min_lr_ratio:.6f}")

        optimizer_for_scheduler = muon if use_muon else adamw
        scheduler = CosineAnnealingLR(
            optimizer=optimizer_for_scheduler,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            max_lr=scaled_lr,
            min_lr_ratio=min_lr_ratio
        )
    else:
        print0(f"  No LR scheduling (constant LR)")

    # Load checkpoint weights and optimizer state if resuming
    if resume_from:
        optimizer = muon if use_muon else adamw
        model, optimizer, metadata = load_checkpoint(resume_from, model=model, optimizer=optimizer)
        if use_muon:
            muon = optimizer
        else:
            adamw = optimizer

        # Load scheduler state if it exists
        if scheduler and 'scheduler_state' in metadata:
            scheduler.load_state_dict(metadata['scheduler_state'])
            print0(f"  ✅ Loaded scheduler state (current LR: {scheduler.get_lr():.6f})")

    # Create data loader
    print0(f"\nSetting up data loader...")
    print0(f"  Batch size: {batch_size}")
    print0(f"  Sequence length: {seq_len}")
    print0(f"  Data source: {'Real (FineWebEdu)' if use_real_data else 'Synthetic'}")

    if use_real_data:
        dataloader = TextDataLoader(
            split="train",
            batch_size=batch_size,
            seq_len=seq_len,
        )
        print0(f"  Vocab size: {dataloader.vocab_size}")
        # Update model vocab size to match tokenizer
        if config.vocab_size != dataloader.vocab_size:
            print0(f"  ⚠️  Model vocab ({config.vocab_size}) != tokenizer vocab ({dataloader.vocab_size})")
            print0(f"  Reinitializing model with correct vocab size...")
            config.vocab_size = dataloader.vocab_size
            model = GPT(config)
            model.init_weights()
            nparams = count_params(model.parameters())
            print0(f"  Updated model parameters: {nparams:,} ({nparams/1e6:.2f}M)")

        # Create validation loader
        val_loader = TextDataLoader(
            split="val",
            batch_size=batch_size,
            seq_len=seq_len,
        )
        print0(f"  Validation batches per eval: {val_batches}")
    else:
        print0(f"  Vocab size: {config.vocab_size}")
        num_batches_per_epoch = 100
        dataloader = SyntheticDataLoader(
            vocab_size=config.vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_batches=num_batches_per_epoch
        )
        # Create synthetic validation loader
        val_loader = SyntheticDataLoader(
            vocab_size=config.vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_batches=val_batches
        )

    # Initialize metrics logger
    metrics_log_file = os.path.join(checkpoint_dir, "metrics.json") if checkpoint_dir else "metrics.json"
    metrics_logger = MetricsLogger(metrics_log_file)
    print0(f"  Metrics will be logged to: {metrics_log_file}")

    # Load token_bytes for BPB calculation
    token_bytes = None
    if use_real_data:
        token_bytes = load_token_bytes()
        if token_bytes is not None:
            print0(f"  ✅ Loaded token_bytes for BPB calculation")

    # Training loop
    print0("\n" + "=" * 70)
    print0("Starting training...")
    print0("=" * 70)

    # Checkpoint setup
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print0(f"Checkpoints will be saved to: {checkpoint_dir}")
        print0(f"Save interval: {save_interval} steps")

    step = start_step
    epoch = 0
    running_loss = 0.0
    best_loss = float('inf')
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

            # Update learning rate
            current_lr = scaled_lr
            if scheduler:
                current_lr = scheduler.step()

            running_loss += loss.item()

            # Logging
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                tokens_per_sec = (batch_size * seq_len) / step_time
                mfu = estimate_mfu(model, batch_size, seq_len, step_time)
                elapsed = time.time() - start_time

                print0(f"Step {step:4d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {step_time*1000:.1f}ms | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"MFU: {mfu:.2f}% | "
                      f"Elapsed: {elapsed:.1f}s")

                running_loss = 0.0

            # Validation
            val_loss = None
            bpb = None
            if step % eval_interval == 0:
                print0(f"\n{'='*70}")
                print0(f"Running validation at step {step}...")
                val_start = time.time()
                val_loss = validate(model, val_loader, num_batches=val_batches)
                val_time = time.time() - val_start
                print0(f"Validation loss: {val_loss:.4f} (took {val_time:.1f}s)")

                # Compute BPB
                bpb = compute_bpb(val_loss, token_bytes)
                print0(f"Bits per byte (BPB): {bpb:.4f}")
                print0(f"{'='*70}\n")

                # Log metrics
                metrics_logger.log(
                    step=step,
                    train_loss=avg_loss if step % log_interval == 0 else None,
                    val_loss=val_loss,
                    bpb=bpb,
                    lr=current_lr
                )

            # Save checkpoint
            if checkpoint_dir and step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.npz")
                optimizer = muon if use_muon else adamw

                # Prepare metadata with scheduler state
                meta = {}
                if scheduler:
                    meta['scheduler_state'] = scheduler.state_dict()
                    meta['current_lr'] = current_lr

                save_checkpoint(model, optimizer, step, loss.item(), checkpoint_path, metadata=meta)

            # Save best checkpoint based on validation loss (if available)
            if checkpoint_dir and val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best.npz")
                optimizer = muon if use_muon else adamw
                best_meta = {"best_val_loss": best_loss, "best_bpb": bpb}
                if scheduler:
                    best_meta['scheduler_state'] = scheduler.state_dict()
                    best_meta['current_lr'] = current_lr
                save_checkpoint(model, optimizer, step, val_loss, best_path, metadata=best_meta)
                print0(f"  🌟 New best checkpoint saved! Val loss: {best_loss:.4f}, BPB: {bpb:.4f}")

            # Stop if max steps reached
            if step >= max_steps:
                break

    # Final checkpoint
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, f"final_step_{step}.npz")
        optimizer = muon if use_muon else adamw
        final_meta = {"final": True}
        if scheduler:
            final_meta['scheduler_state'] = scheduler.state_dict()
        save_checkpoint(model, optimizer, step, loss.item(), final_path, metadata=final_meta)

    # Final summary
    total_time = time.time() - start_time
    print0("\n" + "=" * 70)
    print0("Training complete!")
    print0("=" * 70)
    print0(f"Total steps: {step}")
    print0(f"Total time: {total_time:.1f}s")
    print0(f"Avg time per step: {total_time/(step-start_step)*1000:.1f}ms")
    if checkpoint_dir:
        print0(f"Checkpoints saved to: {checkpoint_dir}")
        print0(f"Best loss: {best_loss:.4f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train nanochat model on MLX")
    parser.add_argument("--model-size", type=str, default="d6", choices=['d2', 'd6', 'd10', 'd14', 'd20'],
                       help="Model size (d2, d6, d10, d14, d20)")
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
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use real data from parquet files (FineWebEdu)")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Steps between logging")
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Steps between validation (default: 500)")
    parser.add_argument("--val-batches", type=int, default=100,
                       help="Number of validation batches to run (default: 100)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory to save checkpoints (default: no saving)")
    parser.add_argument("--save-interval", type=int, default=1000,
                       help="Steps between checkpoint saves (default: 1000)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint to resume from (default: fresh start)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                       help="Number of warmup steps for LR schedule (default: 0 = no warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="Minimum LR as fraction of max LR (default: 0.1)")

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
        use_real_data=args.use_real_data,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        val_batches=args.val_batches,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        resume_from=args.resume_from,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
