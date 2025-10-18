#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for MLX
Trains a pretrained model on instruction-following data

Usage:
    python scripts/sft_mlx.py --base-checkpoint checkpoints/d10_pretrain_causal_fix/best.npz
"""
import os
import sys
import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW
from nanochat.common_mlx import print0
from nanochat.checkpoint_mlx import save_checkpoint, load_checkpoint
from nanochat.lr_scheduler_mlx import CosineAnnealingLR
from nanochat.tokenizer import RustBPETokenizer


def get_tokenizer_mlx():
    """Load tokenizer for MLX (without torch dependency)"""
    # Hardcoded path to tokenizer directory
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    if not os.path.exists(tokenizer_dir):
        # Try current directory
        tokenizer_dir = "tokenizer"
    return RustBPETokenizer.from_directory(tokenizer_dir)


def load_alpaca_dataset():
    """
    Load Alpaca instruction dataset from HuggingFace
    Returns list of instruction/input/output dicts
    """
    try:
        from datasets import load_dataset
        print0("Loading Alpaca dataset from HuggingFace...")

        # Load Stanford Alpaca dataset (52K instruction-following examples)
        ds = load_dataset("tatsu-lab/alpaca", split="train")

        # Convert to list of dicts
        data = []
        for item in ds:
            data.append({
                "instruction": item["instruction"],
                "input": item["input"] if item["input"] else "",
                "output": item["output"]
            })

        print0(f"✅ Loaded {len(data):,} instruction examples")
        return data

    except Exception as e:
        print0(f"⚠️  Failed to load Alpaca dataset: {e}")
        print0("Falling back to synthetic examples...")

        # Fallback: synthetic instruction data
        data = [
            {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
            {"instruction": "Explain machine learning in simple terms.", "input": "", "output": "Machine learning is a type of artificial intelligence where computers learn from data to make predictions or decisions without being explicitly programmed."},
            {"instruction": "Write a haiku about nature.", "input": "", "output": "Spring cherry blossoms\nGentle breeze whispers softly\nNature's art unfolds"},
            {"instruction": "Translate to Spanish", "input": "Hello, how are you?", "output": "Hola, ¿cómo estás?"},
            {"instruction": "What is 15 times 23?", "input": "", "output": "15 times 23 equals 345."},
        ] * 100  # Repeat to make ~500 examples

        print0(f"Using {len(data)} synthetic examples")
        return data


def format_instruction(example, tokenizer):
    """
    Format an instruction example into tokenized input/target pairs using tokenizer's render_conversation

    Format:
        <|bos|><|user_start|>instruction [+ input]<|user_end|><|assistant_start|>output<|assistant_end|>

    Returns:
        input_ids: tokens for input (all tokens except last)
        target_ids: tokens for targets (all tokens shifted by 1)
        loss_mask: 1 where we compute loss (only on assistant response), 0 elsewhere
    """
    # Build prompt
    if example["input"]:
        prompt = f"{example['instruction']}\n\n{example['input']}"
    else:
        prompt = example["instruction"]

    response = example["output"]

    # Create conversation in the format expected by tokenizer
    conversation = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

    # Use tokenizer's render_conversation which already handles masking
    all_tokens, loss_mask = tokenizer.render_conversation(conversation, max_tokens=2048)

    # Input is all tokens except last, target is all tokens except first
    input_ids = all_tokens[:-1]
    target_ids = all_tokens[1:]
    loss_mask = loss_mask[1:]  # Shift mask to align with targets

    return input_ids, target_ids, loss_mask


def create_batches(data, tokenizer, batch_size, max_seq_len=512):
    """
    Create batches from instruction data with padding and masking

    Yields:
        x: input tokens (batch_size, seq_len)
        y: target tokens (batch_size, seq_len)
        mask: loss mask (batch_size, seq_len) - 1 where we compute loss, 0 where we ignore
    """
    # Use assistant_end token as padding (it's masked anyway)
    pad_token = tokenizer.encode_special("<|assistant_end|>")

    batch_inputs = []
    batch_targets = []
    batch_masks = []

    for example in data:
        input_ids, target_ids, loss_mask = format_instruction(example, tokenizer)

        # Truncate if too long
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            target_ids = target_ids[:max_seq_len]
            loss_mask = loss_mask[:max_seq_len]

        batch_inputs.append(input_ids)
        batch_targets.append(target_ids)
        batch_masks.append(loss_mask)

        if len(batch_inputs) == batch_size:
            # Pad batch to same length
            max_len = max(len(ids) for ids in batch_inputs)

            x_batch = []
            y_batch = []
            mask_batch = []

            for inp, tgt, msk in zip(batch_inputs, batch_targets, batch_masks):
                pad_len = max_len - len(inp)
                x_batch.append(inp + [pad_token] * pad_len)
                y_batch.append(tgt + [pad_token] * pad_len)
                mask_batch.append(msk + [0] * pad_len)  # Mask out padding

            yield (
                mx.array(x_batch, dtype=mx.int32),
                mx.array(y_batch, dtype=mx.int32),
                mx.array(mask_batch, dtype=mx.float32)
            )

            batch_inputs = []
            batch_targets = []
            batch_masks = []


def masked_loss_fn(model, x, y, mask):
    """
    Compute masked cross-entropy loss
    Only compute loss where mask == 1
    """
    # Get logits
    logits = model(x)  # (batch, seq_len, vocab_size)

    # Compute cross-entropy
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Cross-entropy loss
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)

    # Apply mask
    masked_losses = losses * mask_flat

    # Average over non-masked tokens
    num_active = mx.sum(mask_flat)
    loss = mx.sum(masked_losses) / mx.maximum(num_active, mx.array(1.0))

    return loss


def train_sft(
    base_checkpoint,
    output_dir="checkpoints/d10_sft",
    batch_size=4,
    max_seq_len=512,
    max_steps=3000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_interval=200,
    save_interval=500,
    log_interval=10,
):
    """
    Run supervised fine-tuning
    """
    print0("=" * 70)
    print0("MLX Supervised Fine-Tuning")
    print0("=" * 70)

    # Load tokenizer
    print0("\nLoading tokenizer...")
    tokenizer = get_tokenizer_mlx()
    print0(f"✅ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size():,})")

    # Load base model
    print0(f"\nLoading base checkpoint: {base_checkpoint}")

    # Load metadata to get config
    meta_path = base_checkpoint + ".meta.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Create model with config from checkpoint
    config_dict = metadata['model_config']
    config = GPTConfig(**config_dict)
    model = GPT(config)

    print0(f"  Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")

    # Load weights
    model, _, load_meta = load_checkpoint(base_checkpoint, model=model, optimizer=None)
    print0(f"  ✅ Loaded weights from step {load_meta['step']}")

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
    print0(f"  Model parameters: {nparams:,} ({nparams/1e6:.2f}M)")

    # Load instruction dataset
    print0("\nLoading instruction dataset...")
    data = load_alpaca_dataset()

    # Split into train/val (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print0(f"  Train examples: {len(train_data):,}")
    print0(f"  Val examples: {len(val_data):,}")
    print0(f"  Dataset loaded successfully")

    # Setup optimizer
    print0(f"\nSetting up optimizer...")
    print0(f"  Learning rate: {learning_rate}")
    print0(f"  Weight decay: {weight_decay}")
    print0(f"  Warmup steps: {warmup_steps}")

    optimizer = AdamW(
        learning_rate=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )

    # LR scheduler
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=learning_rate,
        min_lr_ratio=0.1
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print0(f"  Checkpoints will be saved to: {output_dir}")

    # Training loop
    print0("\n" + "=" * 70)
    print0("Starting SFT training...")
    print0("=" * 70)
    print0(f"Batch size: {batch_size}, Max steps: {max_steps}")
    print0(f"Learning rate: {learning_rate}, Warmup: {warmup_steps}")
    print0("=" * 70)

    step = 0
    epoch = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    start_time = time.time()
    print0("Entering training loop...")

    # Training function with gradient
    def loss_and_grad_fn(model, x, y, mask):
        return mx.value_and_grad(masked_loss_fn)(model, x, y, mask)

    while step < max_steps:
        epoch += 1

        # Shuffle data each epoch
        import random
        random.shuffle(train_data)

        # Create batches
        batch_gen = create_batches(train_data, tokenizer, batch_size, max_seq_len)

        for x, y, mask in batch_gen:
            step += 1

            # Training step
            step_start = time.time()
            loss, grads = loss_and_grad_fn(model, x, y, mask)

            # Update model
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            mx.eval(loss)

            step_time = time.time() - step_start

            # Update learning rate
            current_lr = scheduler.step()

            running_loss += loss.item()

            # Logging
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (batch_size * x.shape[1]) / step_time

                print0(f"Step {step:4d}/{max_steps:4d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {step_time*1000:.1f}ms | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"Elapsed: {elapsed:.1f}s")

                running_loss = 0.0

            # Validation
            if step % eval_interval == 0:
                print0(f"\n{'='*70}")
                print0(f"Running validation at step {step}...")

                # Compute validation loss
                val_losses = []
                val_batch_gen = create_batches(val_data[:500], tokenizer, batch_size, max_seq_len)  # Use subset for speed

                for val_x, val_y, val_mask in val_batch_gen:
                    val_loss = masked_loss_fn(model, val_x, val_y, val_mask)
                    mx.eval(val_loss)
                    val_losses.append(val_loss.item())

                val_loss = sum(val_losses) / len(val_losses)
                print0(f"Validation loss: {val_loss:.4f}")
                print0(f"{'='*70}\n")

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(output_dir, "best.npz")
                    save_checkpoint(model, optimizer, step, val_loss, best_path,
                                  metadata={"val_loss": val_loss, "scheduler_state": scheduler.state_dict()})
                    print0(f"  🌟 New best checkpoint! Val loss: {val_loss:.4f}")

            # Save periodic checkpoint
            if step % save_interval == 0:
                ckpt_path = os.path.join(output_dir, f"step_{step}.npz")
                save_checkpoint(model, optimizer, step, loss.item(), ckpt_path,
                              metadata={"scheduler_state": scheduler.state_dict()})

            if step >= max_steps:
                break

        if step >= max_steps:
            break

    # Save final checkpoint
    final_path = os.path.join(output_dir, f"final_step_{step}.npz")
    save_checkpoint(model, optimizer, step, loss.item(), final_path,
                   metadata={"final": True, "scheduler_state": scheduler.state_dict()})

    # Summary
    total_time = time.time() - start_time
    print0("\n" + "=" * 70)
    print0("SFT Training complete!")
    print0("=" * 70)
    print0(f"Total steps: {step}")
    print0(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print0(f"Best val loss: {best_val_loss:.4f}")
    print0(f"Checkpoints saved to: {output_dir}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for nanochat on MLX")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints/d10_sft",
                       help="Output directory for SFT checkpoints")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--max-steps", type=int, default=3000,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="LR warmup steps")
    parser.add_argument("--eval-interval", type=int, default=200,
                       help="Steps between validation")
    parser.add_argument("--save-interval", type=int, default=500,
                       help="Steps between checkpoint saves")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Steps between logging")

    args = parser.parse_args()

    train_sft(
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )
