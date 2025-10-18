"""
Validation and metrics utilities for MLX training
"""
import json
import os
import time
from pathlib import Path
import mlx.core as mx
from nanochat.common_mlx import print0, get_base_dir


def validate(model, val_loader, num_batches=100):
    """
    Run validation and return average loss

    Args:
        model: GPT model
        val_loader: Validation data loader
        num_batches: Number of batches to validate on

    Returns:
        avg_loss: Average validation loss
    """
    total_loss = 0.0
    count = 0

    # Iterate through validation batches
    for x, y in val_loader:
        if count >= num_batches:
            break

        # Compute loss (no gradients needed)
        loss = model(x, targets=y)
        mx.eval(loss)
        total_loss += loss.item()
        count += 1

    if count == 0:
        return 0.0

    return total_loss / count


def compute_bpb(loss, token_bytes=None):
    """
    Convert cross-entropy loss to bits per byte (BPB)

    BPB is a more interpretable metric than raw loss, especially for text.
    It measures how many bits are needed to encode each byte of text.

    Args:
        loss: Cross-entropy loss (nats)
        token_bytes: Optional array of bytes per token (from tokenizer cache)
                    If None, uses a default estimate

    Returns:
        bpb: Bits per byte
    """
    # Convert nats to bits: divide by ln(2)
    bits_per_token = loss / 0.693147

    # If we have token_bytes, use the mean bytes per token
    if token_bytes is not None:
        if isinstance(token_bytes, mx.array):
            mean_bytes_per_token = mx.mean(token_bytes).item()
        else:
            mean_bytes_per_token = float(mx.mean(mx.array(token_bytes)))
    else:
        # Default estimate: ~4 bytes per token (common for BPE tokenizers)
        mean_bytes_per_token = 4.0

    # BPB = bits per token / bytes per token
    bpb = bits_per_token / mean_bytes_per_token

    return bpb


def load_token_bytes():
    """
    Load token_bytes from tokenizer cache for BPB calculation

    Returns:
        token_bytes: Array of bytes per token, or None if not found
    """
    try:
        base_dir = get_base_dir()
        token_bytes_path = os.path.join(base_dir, "tokenizer", "token_bytes.bin")

        if os.path.exists(token_bytes_path):
            import numpy as np
            token_bytes = np.fromfile(token_bytes_path, dtype=np.uint8)
            return mx.array(token_bytes, dtype=mx.float32)
        else:
            print0(f"⚠️  token_bytes.bin not found at {token_bytes_path}")
            print0(f"   Using default estimate for BPB calculation")
            return None
    except Exception as e:
        print0(f"⚠️  Error loading token_bytes: {e}")
        return None


class MetricsLogger:
    """
    Logger for training metrics with JSON persistence
    """

    def __init__(self, log_file="metrics.json"):
        """
        Args:
            log_file: Path to JSON file for metrics logging
        """
        self.log_file = log_file
        self.metrics = []

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

        # Load existing metrics if file exists
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    self.metrics = json.load(f)
                print0(f"📊 Loaded {len(self.metrics)} existing metric entries from {log_file}")
            except Exception as e:
                print0(f"⚠️  Could not load existing metrics: {e}")
                self.metrics = []

    def log(self, step, train_loss=None, val_loss=None, bpb=None, lr=None, **kwargs):
        """
        Log metrics for a training step

        Args:
            step: Training step number
            train_loss: Training loss (optional)
            val_loss: Validation loss (optional)
            bpb: Bits per byte (optional)
            lr: Learning rate (optional)
            **kwargs: Additional metrics to log
        """
        entry = {
            "step": step,
            "timestamp": time.time(),
        }

        if train_loss is not None:
            entry["train_loss"] = float(train_loss)
        if val_loss is not None:
            entry["val_loss"] = float(val_loss)
        if bpb is not None:
            entry["bpb"] = float(bpb)
        if lr is not None:
            entry["lr"] = float(lr)

        # Add any additional metrics
        for key, value in kwargs.items():
            entry[key] = float(value) if isinstance(value, (int, float)) else value

        self.metrics.append(entry)
        self._save()

    def _save(self):
        """Save metrics to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print0(f"⚠️  Error saving metrics: {e}")

    def get_best_val_loss(self):
        """
        Get the best (lowest) validation loss from logged metrics

        Returns:
            best_val_loss: Lowest validation loss, or inf if no val losses logged
        """
        val_losses = [m["val_loss"] for m in self.metrics if "val_loss" in m]
        return min(val_losses) if val_losses else float('inf')

    def get_latest_metrics(self):
        """
        Get the most recent metric entry

        Returns:
            latest: Dict of latest metrics, or None if no metrics logged
        """
        return self.metrics[-1] if self.metrics else None


def print_metrics_summary(metrics_logger):
    """
    Print a summary of training metrics

    Args:
        metrics_logger: MetricsLogger instance
    """
    if not metrics_logger.metrics:
        print0("No metrics logged yet")
        return

    latest = metrics_logger.get_latest_metrics()
    best_val = metrics_logger.get_best_val_loss()

    print0("\n" + "=" * 70)
    print0("Metrics Summary")
    print0("=" * 70)
    print0(f"Latest step: {latest['step']}")

    if "train_loss" in latest:
        print0(f"Latest train loss: {latest['train_loss']:.4f}")
    if "val_loss" in latest:
        print0(f"Latest val loss: {latest['val_loss']:.4f}")
        print0(f"Best val loss: {best_val:.4f}")
    if "bpb" in latest:
        print0(f"Latest BPB: {latest['bpb']:.4f}")
    if "lr" in latest:
        print0(f"Latest LR: {latest['lr']:.6f}")

    print0("=" * 70)
