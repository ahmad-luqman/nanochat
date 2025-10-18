#!/usr/bin/env python3
"""
Test checkpoint save/load for MLX models
"""
import os
import tempfile
import shutil

import mlx.core as mx
from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW, Muon
from nanochat.checkpoint_mlx import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint
)

print("Testing MLX Checkpoint System")
print("=" * 70)

# Create temporary directory for checkpoints
test_dir = tempfile.mkdtemp(prefix="nanochat_test_")
print(f"Test directory: {test_dir}")
print()

try:
    # Create a tiny model
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128
    )
    model = GPT(config)
    model.init_weights()

    print(f"Model: {config.n_layer} layers, {config.n_embd} dim")

    # Test 1: Save checkpoint with AdamW
    print("\n" + "=" * 70)
    print("Test 1: Save checkpoint with AdamW")
    print("=" * 70)

    adamw = AdamW(learning_rate=0.01)

    # Do one training step to initialize optimizer state
    idx = mx.random.randint(0, config.vocab_size, (2, 16))
    targets = mx.random.randint(0, config.vocab_size, (2, 16))

    def loss_fn(model, x, y):
        return model(x, targets=y)

    loss, grads = mx.value_and_grad(loss_fn)(model, idx, targets)
    adamw.update(model, grads)
    mx.eval(model.parameters(), loss)

    initial_loss = loss.item()
    print(f"Initial loss: {initial_loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(test_dir, "checkpoint_test1.npz")
    save_checkpoint(model, adamw, step=1, loss=loss, save_path=checkpoint_path)

    # Verify files exist
    assert os.path.exists(checkpoint_path), "Checkpoint file not found"
    assert os.path.exists(checkpoint_path + ".opt.npz"), "Optimizer state not found"
    assert os.path.exists(checkpoint_path + ".meta.json"), "Metadata not found"
    print("✅ All checkpoint files created")

    # Test 2: Load checkpoint
    print("\n" + "=" * 70)
    print("Test 2: Load checkpoint")
    print("=" * 70)

    # Create new model and optimizer
    model2 = GPT(config)
    model2.init_weights()  # Initialize with different weights
    adamw2 = AdamW(learning_rate=0.01)

    # Verify weights are different before loading
    w1 = model.wte.weight
    w2_before = model2.wte.weight
    diff_before = mx.sum(mx.abs(w1 - w2_before)).item()
    print(f"Weight difference before load: {diff_before:.2f}")
    assert diff_before > 0, "Weights should be different before loading"

    # Load checkpoint
    model2, adamw2, meta = load_checkpoint(checkpoint_path, model2, adamw2)

    print(f"Loaded checkpoint from step {meta['step']}")
    assert meta['step'] == 1
    assert abs(meta['loss'] - initial_loss) < 1e-5

    # Verify weights are now identical
    w2_after = model2.wte.weight
    diff_after = mx.sum(mx.abs(w1 - w2_after)).item()
    print(f"Weight difference after load: {diff_after:.6f}")
    assert diff_after < 1e-5, "Weights should be identical after loading"

    print("✅ Checkpoint loaded correctly")

    # Test 3: Training continuation
    print("\n" + "=" * 70)
    print("Test 3: Training continuation")
    print("=" * 70)

    # Continue training from checkpoint
    losses = []
    for step in range(5):
        loss, grads = mx.value_and_grad(loss_fn)(model2, idx, targets)
        adamw2.update(model2, grads)
        mx.eval(model2.parameters(), loss)
        losses.append(loss.item())
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print(f"Loss decreased: {losses[0] > losses[-1]} ({'✅' if losses[0] > losses[-1] else '❌'})")

    # Save another checkpoint
    checkpoint_path2 = os.path.join(test_dir, "checkpoint_test2.npz")
    save_checkpoint(model2, adamw2, step=6, loss=losses[-1], save_path=checkpoint_path2)
    print("✅ Training continuation successful")

    # Test 4: Multiple checkpoints
    print("\n" + "=" * 70)
    print("Test 4: List and find checkpoints")
    print("=" * 70)

    checkpoints = list_checkpoints(test_dir)
    print(f"Found {len(checkpoints)} checkpoints:")
    for path, step, loss in checkpoints:
        print(f"  Step {step}: {os.path.basename(path)} (loss={loss:.4f})")

    assert len(checkpoints) == 2, "Should have 2 checkpoints"

    latest = get_latest_checkpoint(test_dir)
    print(f"\nLatest checkpoint: {os.path.basename(latest)}")
    assert latest == checkpoint_path2, "Latest should be checkpoint_test2"
    print("✅ Checkpoint listing works")

    # Test 5: Checkpoint with Muon optimizer
    print("\n" + "=" * 70)
    print("Test 5: Save/load with Muon optimizer")
    print("=" * 70)

    model3 = GPT(config)
    model3.init_weights()
    muon = Muon(learning_rate=0.02, momentum=0.95)

    # Train one step
    loss, grads = mx.value_and_grad(loss_fn)(model3, idx, targets)
    muon.update(model3, grads)
    mx.eval(model3.parameters(), loss)

    # Save
    checkpoint_path3 = os.path.join(test_dir, "checkpoint_muon.npz")
    save_checkpoint(model3, muon, step=1, loss=loss, save_path=checkpoint_path3)

    # Load into new model
    model4 = GPT(config)
    model4.init_weights()
    muon2 = Muon(learning_rate=0.02, momentum=0.95)

    model4, muon2, meta = load_checkpoint(checkpoint_path3, model4, muon2)

    # Verify
    w3 = model3.wte.weight
    w4 = model4.wte.weight
    diff = mx.sum(mx.abs(w3 - w4)).item()
    print(f"Weight difference: {diff:.6f}")
    assert diff < 1e-5, "Muon checkpoint should load correctly"
    print("✅ Muon checkpoint works")

    # Test 6: Metadata preservation
    print("\n" + "=" * 70)
    print("Test 6: Metadata preservation")
    print("=" * 70)

    custom_meta = {
        "run_name": "test_run",
        "learning_rate": 0.01,
        "batch_size": 8
    }

    checkpoint_path_meta = os.path.join(test_dir, "checkpoint_meta.npz")
    save_checkpoint(
        model, adamw, step=100, loss=2.5,
        save_path=checkpoint_path_meta,
        metadata=custom_meta
    )

    _, _, meta = load_checkpoint(checkpoint_path_meta)

    assert meta["run_name"] == "test_run"
    assert meta["learning_rate"] == 0.01
    assert meta["batch_size"] == 8
    assert meta["step"] == 100
    print("✅ Custom metadata preserved")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ Save checkpoint with AdamW")
    print("✅ Load checkpoint correctly")
    print("✅ Training continuation works")
    print("✅ List and find checkpoints")
    print("✅ Muon optimizer checkpoint")
    print("✅ Custom metadata preservation")
    print("\nAll checkpoint tests passed!")

finally:
    # Cleanup
    print(f"\nCleaning up test directory: {test_dir}")
    shutil.rmtree(test_dir)
