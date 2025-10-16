#!/usr/bin/env python3
"""
Exercise 4.4: Checkpointing - Save and Restore Model State

Learn how to:
- Save model checkpoints during training
- Load checkpoints to resume training
- Understand why checkpointing is crucial
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.checkpoint_mlx import save_checkpoint, load_checkpoint
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import mlx.nn as nn
import os
import tempfile

print("=" * 60)
print("Checkpointing: Save and Restore Model State")
print("=" * 60)

# Create a temporary directory for checkpoints
temp_dir = tempfile.mkdtemp()
checkpoint_dir = os.path.join(temp_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"\nCheckpoint directory: {checkpoint_dir}\n")

# Create and train a tiny model
config = GPTConfig(
    sequence_len=32,
    vocab_size=100,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=64
)
model = GPT(config)
model.init_weights()

optimizer = AdamW(learning_rate=0.01)

print("Step 1: Train a model for 50 steps")
print("-" * 60)

def loss_fn(model, x, y):
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = y.reshape(-1)
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)
    return mx.mean(losses)

train_data = mx.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=mx.int32)

# Get initial prediction
test_input = mx.array([[1, 2, 3]], dtype=mx.int32)
initial_logits = model(test_input)
initial_pred = mx.argmax(initial_logits[0, -1, :]).item()
print(f"Initial prediction for [1,2,3]: {initial_pred}")

# Train for 50 steps
print("\nTraining...")
for step in range(50):
    inputs = train_data[:, :-1]
    targets = train_data[:, 1:]
    loss_val, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    mx.eval(loss_val)

    if step % 10 == 0:
        print(f"  Step {step:3d}: Loss = {loss_val.item():.4f}")

# Check trained prediction
trained_logits = model(test_input)
trained_pred = mx.argmax(trained_logits[0, -1, :]).item()
print(f"\nTrained prediction for [1,2,3]: {trained_pred}")

print("\nStep 2: Save checkpoint at step 50")
print("-" * 60)

checkpoint_path = os.path.join(checkpoint_dir, "model_step_50.npz")
save_checkpoint(
    model=model,
    optimizer=optimizer,
    step=50,
    loss=loss_val.item(),
    checkpoint_path=checkpoint_path,
    metadata={"dataset": "simple_sequence", "pattern": "1,2,3,4,5"}
)

print(f"✓ Checkpoint saved to: {checkpoint_path}")

# Check files
files = os.listdir(checkpoint_dir)
print(f"\nFiles created:")
for f in files:
    fpath = os.path.join(checkpoint_dir, f)
    fsize = os.path.getsize(fpath) / 1024  # KB
    print(f"  - {f} ({fsize:.1f} KB)")

print("\nStep 3: Create a NEW model with random weights")
print("-" * 60)

new_model = GPT(config)
new_model.init_weights()

random_logits = new_model(test_input)
random_pred = mx.argmax(random_logits[0, -1, :]).item()
print(f"Random model prediction for [1,2,3]: {random_pred}")
print(f"  (Very different from trained: {trained_pred})")

print("\nStep 4: Load checkpoint into new model")
print("-" * 60)

print(f"Loading checkpoint...")
loaded_model, loaded_optimizer, metadata = load_checkpoint(
    checkpoint_path,
    model=new_model,
    optimizer=optimizer
)

print(f"✓ Checkpoint loaded!")
print(f"  Step: {metadata['step']}")
print(f"  Loss: {metadata['loss']:.4f}")
print(f"  Metadata: {metadata.get('metadata', {})}")

# Test loaded model
loaded_logits = loaded_model(test_input)
loaded_pred = mx.argmax(loaded_logits[0, -1, :]).item()
print(f"\nLoaded model prediction for [1,2,3]: {loaded_pred}")
print(f"  Matches trained? {initial_pred == loaded_pred}")

print(f"\n{'='*60}")
print("Verification")
print(f"{'='*60}")

print(f"\nComparing all three models:")
print(f"  Initial (before training): {initial_pred}")
print(f"  After training:            {trained_pred}")
print(f"  Random new model:          {random_pred}")
print(f"  Loaded from checkpoint:    {loaded_pred}")

if loaded_pred == trained_pred:
    print(f"\n✓ SUCCESS! Checkpoint correctly restored model state!")
else:
    print(f"\n✗ WARNING: Predictions don't match!")

print(f"\n{'='*60}")
print("Why Checkpointing is Critical")
print(f"{'='*60}")

print("""
1. INTERRUPTED TRAINING:
   - Training a large model takes days/weeks
   - If interrupted: load checkpoint and continue
   - Without checkpoints: restart from zero!

2. HYPERPARAMETER TUNING:
   - Try different learning rates from checkpoint
   - Compare different training schedules
   - Reset to good checkpoint if experiments fail

3. ENSEMBLE MODELS:
   - Save multiple checkpoints
   - Combine predictions from different points

4. BEST MODEL TRACKING:
   - Save checkpoint with best validation loss
   - Use that for final deployment
   - Not just the last checkpoint

5. DEBUGGING & ANALYSIS:
   - Analyze model weights at different training stages
   - Track how specific parameters change
   - Understand what the model learned
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")
print("""
1. Train model 1, save checkpoint
2. Load checkpoint into model 2
3. Continue training for 50 more steps
4. Verify loss continues to decrease (no reset)
5. Save checkpoints every 10 steps, compare all of them
6. Load different checkpoints and see evolution of predictions
7. Implement "save best checkpoint" logic
8. Clean up: How much disk space for 1000 checkpoints?
""")
