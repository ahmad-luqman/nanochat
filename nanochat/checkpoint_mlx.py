"""
Checkpoint management for MLX models
Save and load model/optimizer state for training continuation
"""
import os
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn


def save_checkpoint(model, optimizer, step, loss, save_path, metadata=None):
    """
    Save model and optimizer checkpoint to disk.

    Args:
        model: GPT model instance
        optimizer: Optimizer instance (AdamW or Muon)
        step: Current training step
        loss: Current loss value
        save_path: Path to save checkpoint (e.g., "out/checkpoint_1000.npz")
        metadata: Optional dict of additional metadata

    Saves:
        - {save_path} - Model weights (NPZ format)
        - {save_path}.opt.npz - Optimizer state
        - {save_path}.meta.json - Metadata (step, loss, config, timestamp)
    """
    # Create output directory
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    print(f"Saving checkpoint to {save_path}...")
    weights = model.parameters()
    mx.savez(save_path, **flatten_dict(weights))

    # Save optimizer state
    opt_path = save_path + ".opt.npz"
    opt_state = optimizer.state
    if opt_state:
        mx.savez(opt_path, **flatten_dict(opt_state))
        print(f"  Saved optimizer state to {opt_path}")

    # Save metadata
    meta_path = save_path + ".meta.json"
    meta = {
        "step": step,
        "loss": float(loss),
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "sequence_len": model.config.sequence_len,
            "vocab_size": model.config.vocab_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
        },
        "optimizer_type": optimizer.__class__.__name__,
    }
    if metadata:
        meta.update(metadata)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")
    print(f"✅ Checkpoint saved at step {step}, loss {loss:.4f}")


def load_checkpoint(checkpoint_path, model=None, optimizer=None):
    """
    Load model and optimizer checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional GPT model instance to load weights into
        optimizer: Optional optimizer instance to load state into

    Returns:
        model: Model with loaded weights (or None if model not provided)
        optimizer: Optimizer with loaded state (or None if optimizer not provided)
        metadata: Dict of checkpoint metadata

    If model/optimizer not provided, only metadata is returned.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load metadata
    meta_path = checkpoint_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        print(f"  Checkpoint from step {metadata['step']}, loss {metadata['loss']:.4f}")
    else:
        metadata = {}
        print(f"  Warning: No metadata file found")

    # Load model weights
    if model is not None:
        weights = mx.load(checkpoint_path)
        weights = unflatten_dict(weights)
        # Use update() to load weights into model
        model.update(weights)
        print(f"  ✅ Loaded model weights")

    # Load optimizer state
    if optimizer is not None:
        opt_path = checkpoint_path + ".opt.npz"
        if os.path.exists(opt_path):
            opt_state = mx.load(opt_path)
            opt_state = unflatten_dict(opt_state)
            optimizer.state = opt_state
            print(f"  ✅ Loaded optimizer state")
        else:
            print(f"  Warning: No optimizer state file found")

    return model, optimizer, metadata


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dict to single level with dot-separated keys.

    Example:
        {'h': [{'attn': {'weight': x}}, {'attn': {'weight': y}}]}
        → {'h.0.attn.weight': x, 'h.1.attn.weight': y}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({str(i): item}, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='.'):
    """
    Unflatten dict with dot-separated keys back to nested structure.

    Example:
        {'h.0.attn.weight': x, 'h.1.attn.weight': y}
        → {'h': [{'attn': {'weight': x}}, {'attn': {'weight': y}}]}
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result

        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            # Check if next part is a number (indicates list)
            next_part = parts[i + 1]
            is_list = next_part.isdigit()

            if part.isdigit():
                # Current part is list index
                idx = int(part)
                if not isinstance(current, list):
                    current = []
                # Extend list if needed
                while len(current) <= idx:
                    current.append({} if not is_list else [])
                current = current[idx]
            else:
                # Current part is dict key
                if part not in current:
                    current[part] = [] if is_list else {}
                current = current[part]

        # Set the final value
        final_key = parts[-1]
        if final_key.isdigit():
            idx = int(final_key)
            if not isinstance(current, list):
                current = []
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[final_key] = value

    return result


def list_checkpoints(checkpoint_dir):
    """
    List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint paths sorted by step number
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".npz") and not file.endswith(".opt.npz"):
            path = os.path.join(checkpoint_dir, file)
            meta_path = path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                checkpoints.append((path, meta["step"], meta["loss"]))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None
    return checkpoints[-1][0]
