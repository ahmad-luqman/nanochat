# Plan B - Advanced MLX Features

**Status:** In Progress
**Estimated Time:** 2-4 hours
**Goal:** Add production-ready features to the MLX training pipeline

## Prerequisites
- ✅ Plan A completed (core training pipeline working)
- ✅ Tokenizer trained (65K vocab)
- ✅ 8 data shards downloaded (~720MB)

## Phase B.1: Checkpoint Management (30-45 min)

### Tasks:
1. **Add checkpoint saving to training loop**
   - Save every N steps (configurable)
   - Save best checkpoint (lowest validation loss)
   - Include metadata: step, loss, config, timestamp

2. **Implement checkpoint resuming**
   - Load checkpoint and continue training
   - Restore optimizer state
   - Resume from correct step number

3. **Add checkpoint CLI arguments**
   - `--checkpoint-dir`: Directory to save checkpoints
   - `--save-interval`: Steps between saves (default: 1000)
   - `--resume-from`: Path to checkpoint to resume from

### Files to modify:
- `scripts/train_mlx.py`: Add save/resume logic
- `nanochat/checkpoint_mlx.py`: Add optimizer state save/load

### Success criteria:
- [ ] Training saves checkpoints every N steps
- [ ] Can resume training from checkpoint
- [ ] Optimizer state is preserved across resume

---

## Phase B.2: Learning Rate Scheduling (30-45 min)

### Tasks:
1. **Implement Cosine Annealing LR scheduler**
   - Warmup: Linear ramp from 0 to max_lr over warmup_steps
   - Cosine decay: From max_lr to min_lr over remaining steps
   - Min LR: 10% of max_lr (configurable)

2. **Add scheduler to training loop**
   - Update LR every step
   - Log current LR
   - Save LR state in checkpoint

3. **Add scheduler CLI arguments**
   - `--warmup-steps`: Number of warmup steps (default: 1000)
   - `--max-steps`: Total training steps (for cosine schedule)
   - `--min-lr-ratio`: Min LR as fraction of max LR (default: 0.1)

### Implementation:
```python
class CosineAnnealingLR:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self.optimizer.learning_rate = lr
        return lr
```

### Files to create/modify:
- `nanochat/lr_scheduler_mlx.py`: New file with scheduler classes
- `scripts/train_mlx.py`: Integrate scheduler

### Success criteria:
- [ ] LR warms up linearly for warmup_steps
- [ ] LR decays with cosine schedule
- [ ] Current LR is logged and saved in checkpoint

---

## Phase B.3: Validation and Metrics (45-60 min)

### Tasks:
1. **Implement validation loop**
   - Run on validation split every eval_interval steps
   - Compute average loss on N validation batches
   - Track best validation loss

2. **Add bits-per-byte (BPB) metric**
   - Load token_bytes from tokenizer cache
   - Convert cross-entropy loss to bits per byte
   - More interpretable than raw loss

3. **Enhanced logging**
   - Log train loss, val loss, BPB
   - Save metrics to JSON file
   - Optional: TensorBoard/Weights & Biases integration

4. **Early stopping (optional)**
   - Stop if val loss doesn't improve for N eval intervals
   - Prevents overfitting

### Implementation structure:
```python
def validate(model, val_loader, num_batches=100):
    """Run validation and return average loss"""
    total_loss = 0.0
    count = 0

    for x, y in val_loader:
        if count >= num_batches:
            break
        loss = model(x, targets=y)
        mx.eval(loss)
        total_loss += loss.item()
        count += 1

    return total_loss / count

def loss_to_bpb(loss, token_bytes):
    """Convert cross-entropy loss to bits per byte"""
    # BPB = loss / log(2) / mean(token_bytes)
    return loss / 0.693147 / mx.mean(token_bytes).item()
```

### Files to create/modify:
- `nanochat/metrics_mlx.py`: Validation and BPB functions
- `scripts/train_mlx.py`: Add validation loop
- `metrics.json`: Training metrics log (created during training)

### Success criteria:
- [ ] Validation runs every eval_interval steps
- [ ] BPB metric is computed and logged
- [ ] Best checkpoint is saved based on validation loss
- [ ] Metrics are saved to JSON file

---

## Phase B.4: KV Cache for Inference (45-60 min)

### Tasks:
1. **Implement KV cache in attention**
   - Cache key and value tensors from previous tokens
   - Only compute new KV pairs for new tokens
   - Dramatically speeds up autoregressive generation

2. **Update GPT model for caching**
   - Add `use_cache` parameter to forward pass
   - Return cache along with logits
   - Accept previous cache and append to it

3. **Update chat CLI to use cache**
   - Maintain cache across generation loop
   - Reduces redundant computation
   - 5-10x speedup for long sequences

### Implementation:
```python
class CausalSelfAttention(nn.Module):
    def __call__(self, x, cache=None, use_cache=False):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if cache is not None:
            # Append to existing cache
            k = mx.concatenate([cache['k'], k], axis=1)
            v = mx.concatenate([cache['v'], v], axis=1)

        # Attention computation...
        out = self.compute_attention(q, k, v)

        new_cache = None
        if use_cache:
            new_cache = {'k': k, 'v': v}

        return out, new_cache
```

### Files to modify:
- `nanochat/gpt_mlx.py`: Add cache support to attention and GPT
- `scripts/chat_cli_mlx.py`: Use cache during generation

### Success criteria:
- [ ] KV cache is implemented in attention layer
- [ ] Chat generation uses cache
- [ ] 5-10x speedup for multi-turn conversations
- [ ] Output quality is identical with/without cache

---

## Testing Plan

After each phase:
1. Run unit tests
2. Run `test_end_to_end_mlx.sh`
3. Verify metrics are reasonable
4. Check memory usage

Final integration test:
```bash
# Train d6 for 1000 steps with all features
python scripts/train_mlx.py \
    --model-size d6 \
    --batch-size 8 \
    --seq-len 512 \
    --max-steps 1000 \
    --use-real-data \
    --learning-rate 0.001 \
    --warmup-steps 100 \
    --save-interval 250 \
    --checkpoint-dir checkpoints/d6_test \
    --log-interval 50

# Resume training
python scripts/train_mlx.py \
    --model-size d6 \
    --resume-from checkpoints/d6_test/step_1000.npz \
    --max-steps 2000

# Chat with best checkpoint
python scripts/chat_cli_mlx.py \
    --checkpoint checkpoints/d6_test/best.npz \
    --prompt "What is the capital of France?"
```

---

## Expected Outcomes

**Performance improvements:**
- Checkpoint management: Can save/resume long training runs
- LR scheduling: Better convergence, higher final performance
- Validation: Early detection of overfitting
- KV cache: 5-10x faster inference

**Production readiness:**
- Robust training pipeline
- Proper experiment tracking
- Efficient inference
- Ready for larger-scale training

**Metrics after Plan B:**
- d6 model trained for 1K steps: Loss ~6.5, BPB ~1.8
- d10 model trained for 1K steps: Loss ~6.0, BPB ~1.7
- Generation speed with cache: 100-200 tok/s on M4 Max

---

## Time Estimates

| Phase | Task | Time |
|-------|------|------|
| B.1 | Checkpoint management | 30-45 min |
| B.2 | LR scheduling | 30-45 min |
| B.3 | Validation & metrics | 45-60 min |
| B.4 | KV cache | 45-60 min |
| | Testing & integration | 30 min |
| | **Total** | **3-4 hours** |

---

## Next Steps After Plan B

Once Plan B is complete, you'll be ready for:
- **Plan C**: Full training run (d10-d20 models)
- Production deployment
- Fine-tuning on specific tasks
- Scaling to larger models
