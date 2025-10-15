# MLX Port Implementation Plan for nanochat

**Date:** October 15, 2025
**Goal:** Port nanochat to run natively on Apple Silicon using MLX framework
**Approach:** Hybrid - Start with single-device training, expand to full pipeline

---

## Summary of Findings

### Good News
- MLX has PyTorch-like APIs (`mlx.nn`, `mlx.optimizers`)
- MLX supports transformer training with examples
- MLX now has distributed training support (as of Aug 2025)
- MLX unified memory = no device management headaches
- Basic ops like attention, embeddings, linear layers all exist

### Challenges
- No native DDP equivalent (uses MPI or ring backend instead)
- Different data loading patterns (no DataLoader)
- Some PyTorch ops may not have direct equivalents
- Performance on single Mac won't match 8×H100 (expect 50-100x slower)
- Memory constraints: Mac has 32-96GB unified memory vs 640GB GPU VRAM

---

## Implementation Phases

### Phase 1: Core Model Port (2-3 days) ⚠️ PRIORITY 0
**Files to modify:**
- `nanochat/gpt.py` - Main model architecture
- `pyproject.toml` - Update dependencies

**Key changes:**
```python
# Replace:
import torch
import torch.nn as nn
import torch.nn.functional as F

# With:
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
```

**Specific transformations:**

1. **Attention mechanism** (gpt.py:64-126):
   - Replace `F.scaled_dot_product_attention` with MLX equivalent
   - MLX has `mx.fast.scaled_dot_product_attention`
   - Rotary embeddings: adapt to MLX array ops

2. **Activation functions** (gpt.py:137):
   - `F.relu(x).square()` → MLX has `mx.nn.relu`

3. **Normalization** (gpt.py:36-38):
   - Replace `F.rms_norm` with MLX `mx.fast.rms_norm`

4. **Linear layers**: MLX `nn.Linear` is compatible

5. **Embeddings**: MLX `nn.Embedding` is compatible

6. **Data types**:
   - Replace `torch.bfloat16` → `mx.bfloat16`
   - Replace `torch.float32` → `mx.float32`

7. **Tensor operations**:
   - `torch.cat` → `mx.concatenate`
   - `torch.tensor` → `mx.array`
   - `.view()` → `.reshape()`
   - `.size()` → `.shape`
   - `.to()` → `.astype()`

---

### Phase 2: Training Infrastructure (2-3 days) ⚠️ PRIORITY 0
**Files to modify:**
- `nanochat/common.py` - Remove CUDA/DDP code
- `nanochat/engine.py` - Adapt inference engine
- `nanochat/adamw.py` - Port AdamW optimizer
- `nanochat/muon.py` - Port Muon optimizer

**Key changes:**

1. **Remove DDP code** (common.py:78-127):
   ```python
   # Remove: torch.distributed, torch.cuda
   # Replace compute_init() to remove CUDA checks
   # MLX uses unified memory, no device placement needed

   def compute_init_mlx():
       """MLX initialization - no device management needed"""
       mx.random.seed(42)
       # MLX automatically uses unified memory
       return False, 0, 0, 1  # no ddp, rank 0, local_rank 0, world_size 1
   ```

2. **Optimizer ports**:
   - MLX has `mlx.optimizers.AdamW` built-in
   - Need custom Muon implementation (copy logic, use MLX ops)
   - Distributed optimizers not needed initially (single device)

3. **KV Cache** (engine.py:56-124):
   - Replace `torch.empty` → `mx.zeros`
   - Replace `torch.tensor` → `mx.array`
   - Replace `.to()` → `.astype()`
   - Replace tensor operations with MLX equivalents

---

### Phase 3: Data Pipeline (1-2 days) ⚠️ PRIORITY 0
**Files to modify:**
- `nanochat/dataloader.py` - Replace PyTorch DataLoader
- `nanochat/dataset.py` - Already framework-agnostic (keep as is)

**Key changes:**

1. **Create MLX DataLoader**:
   ```python
   # MLX doesn't have DataLoader, create custom iterator
   # Use Python generators + MLX array conversion

   class MLXDataLoader:
       def __init__(self, dataset, batch_size, ...):
           self.dataset = dataset
           self.batch_size = batch_size

       def __iter__(self):
           for batch_data in self.dataset:
               # Convert to MLX arrays
               yield mx.array(batch_data)
   ```

2. **Tokenization**: rustbpe works with any framework (no changes needed)

3. **Batch collation**: Implement custom collate function using numpy/MLX

---

### Phase 4: Training Scripts (2-3 days) ⚠️ PRIORITY 1
**Files to modify:**
- `scripts/base_train.py`
- `scripts/mid_train.py`
- `scripts/chat_sft.py`
- `scripts/chat_rl.py`

**Key changes:**

1. **Remove torchrun**:
   - Single-device initially (no distributed)
   - For distributed later: use `mpirun` with MLX distributed

2. **Training loop adaptations**:
   ```python
   # Replace PyTorch style:
   loss = model(x, targets=y)
   loss.backward()
   optimizer.step()
   optimizer.zero_grad()

   # With MLX style:
   def loss_fn(model, x, y):
       return model(x, targets=y)

   loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
   optimizer.update(model, grads)
   mx.eval(model.parameters())  # materialize lazy computation
   ```

3. **Mixed precision**: MLX handles automatically with unified memory

4. **Gradient accumulation**: Implement manually by accumulating gradients

5. **Checkpointing**: Adapt to use `mx.save` / `mx.load`

---

### Phase 5: Evaluation & Inference (1-2 days) ⚠️ PRIORITY 1
**Files to modify:**
- `scripts/base_eval.py`
- `scripts/chat_eval.py`
- `scripts/chat_cli.py`
- `scripts/chat_web.py`
- `nanochat/core_eval.py`
- `nanochat/loss_eval.py`

**Key changes:**
- Replace `@torch.inference_mode()` → `@mx.compile` (optional for speed)
- Replace `torch.no_grad()` context manager (MLX doesn't need it)
- Adapt generation loops to use MLX arrays
- Web/CLI interfaces mostly unchanged (just tensor→array conversion)
- Evaluation benchmarks should work as-is (mostly numpy-based)

---

### Phase 6: Dependencies & Config (1 day) ⚠️ PRIORITY 2
**Files to modify:**
- `pyproject.toml`
- `speedrun.sh`
- `.python-version` (stays 3.10+)

**Changes to pyproject.toml:**
```toml
[project.dependencies]
# Remove:
# torch>=2.8.0

# Add:
mlx>=0.29.0
mlx-lm>=0.20.0  # helpful utilities

# Keep all framework-agnostic deps:
datasets>=4.0.0
fastapi>=0.117.1
numpy==1.26.4
psutil>=7.1.0
regex>=2025.9.1
tiktoken>=0.11.0
tokenizers>=0.22.0
uvicorn>=0.36.0
wandb>=0.21.3

# Remove CUDA-specific torch index:
# [tool.uv.sources]
# [tool.uv.index]
```

**Changes to speedrun.sh:**
- Remove CUDA-specific environment variables
- Remove `torchrun` → use `python` directly (or `mpirun` for distributed)
- Adjust batch sizes for Mac memory constraints:
  ```bash
  # For d20 on Mac with 64GB RAM:
  --device_batch_size=1  # vs 32 on 8×H100

  # Or train smaller model:
  --depth=10  # ~75M params instead of 561M
  ```
- Reduce data requirements for testing:
  ```bash
  # Download fewer shards for initial testing
  python -m nanochat.dataset -n 10  # vs 240
  ```

---

## Effort Estimates

| Phase | Complexity | Time | Priority |
|-------|-----------|------|----------|
| Phase 1: Core Model | Medium | 2-3 days | P0 (Critical) |
| Phase 2: Training Infra | High | 2-3 days | P0 (Critical) |
| Phase 3: Data Pipeline | Low | 1-2 days | P0 (Critical) |
| Phase 4: Training Scripts | Medium | 2-3 days | P1 (High) |
| Phase 5: Eval/Inference | Low | 1-2 days | P1 (High) |
| Phase 6: Config | Low | 1 day | P2 (Medium) |
| **Total** | | **9-14 days** | |

---

## Risk Assessment

### High Risk ⚠️
1. **Performance**: Single Mac M-series will be 50-100x slower than 8×H100
   - **Mitigation**: Train smaller model (d10-d15 instead of d20), fewer tokens
   - **Reality check**: d10 on Mac ~4-8 hours vs d20 on 8×H100 ~4 hours

2. **Memory constraints**: Mac has 32-96GB unified memory vs 640GB GPU VRAM
   - **Mitigation**: Reduce batch size to 1-4, use gradient accumulation
   - **Adjust expectations**: May need d6-d10 for 32GB Mac, d15-d20 for 96GB Mac

3. **MLX maturity**: Some edge cases may not work
   - **Mitigation**: Test incrementally, have fallback strategies
   - **Contingency**: Keep PyTorch version as reference

### Medium Risk ⚙️
1. **Distributed training**: MLX's distributed support is newer
   - **Mitigation**: Start with single device, add distributed later if needed
   - **Future work**: Can add MPI-based distributed training for multi-Mac setups

2. **Custom ops**: Muon optimizer needs manual port
   - **Mitigation**: Could fallback to AdamW only initially
   - **Impact**: May affect convergence slightly, but acceptable for proof-of-concept

3. **API mismatches**: Some PyTorch ops may behave slightly differently
   - **Mitigation**: Add compatibility layer where needed
   - **Testing**: Validate outputs match PyTorch version on small examples

### Low Risk ✅
1. **Data loading**: Easy to replace with Python generators
2. **Inference**: MLX excels at inference on Mac
3. **Tokenizer**: rustbpe is framework-agnostic (no changes)
4. **Evaluation**: Most benchmarks are numpy-based

---

## Implementation Strategy

### Recommended Approach: **Option C - Hybrid** ✅

**Rationale:**
1. Validates the port incrementally
2. Enables local development/testing with small models
3. Can scale to full port later
4. Falls back to cloud for serious training
5. Best ROI for effort

**What we'll build:**
- ✅ Full model port (inference + training)
- ✅ Single-device training
- ✅ All evaluation benchmarks
- ✅ CLI and Web chat interfaces
- ❌ Distributed training (future work)
- ⚠️ Train smaller models locally (d6-d15)

---

## First Steps (Quick Wins)

### Step 1: Create MLX branch ✅
```bash
git checkout -b mlx-port
```

### Step 2: Port Core Model (Day 1-2)
1. Create `nanochat/gpt_mlx.py` (side-by-side with original)
2. Port basic operations (norm, rotary, attention)
3. Port full model architecture
4. Test forward pass on dummy data

### Step 3: Minimal Training Test (Day 3)
1. Create `scripts/mlx_test_train.py` (minimal script)
2. Port just enough of training infrastructure
3. Train tiny model (d6, 1M params, 10 data shards)
4. Validate loss decreases over 100 steps

### Step 4: Validate Against PyTorch (Day 3)
1. Initialize same random weights in both frameworks
2. Run same forward pass
3. Compare outputs (should match within float precision)
4. Compare gradients (should match within float precision)

### Step 5: Expand to Full Pipeline (Day 4-14)
1. Port all training scripts
2. Port evaluation scripts
3. Port inference scripts
4. Update documentation

---

## Testing Strategy

### Unit Tests
- [ ] Test individual layer ports (Linear, Attention, MLP)
- [ ] Test rotary embeddings match PyTorch
- [ ] Test RMS norm matches PyTorch
- [ ] Test KV cache operations

### Integration Tests
- [ ] Test full forward pass matches PyTorch
- [ ] Test backward pass/gradients match PyTorch
- [ ] Test generation produces reasonable text
- [ ] Test checkpoint save/load

### End-to-End Tests
- [ ] Train d6 for 1000 steps, verify loss decreases
- [ ] Train d10 for 10K steps, verify CORE score > 0.1
- [ ] Run all evaluation benchmarks
- [ ] Test CLI chat interface
- [ ] Test Web chat interface

---

## Success Criteria

### Minimum Viable Port (MVP)
- [ ] Model runs inference on Mac
- [ ] Model can be trained (single device)
- [ ] Loss decreases during training
- [ ] Can generate coherent text
- [ ] CLI chat works

### Full Success
- [ ] All evaluation benchmarks pass
- [ ] d10 model trains in reasonable time (~8 hours)
- [ ] CORE score comparable to PyTorch version
- [ ] Web UI works
- [ ] Documentation updated

### Stretch Goals
- [ ] Distributed training across multiple Macs
- [ ] Optimized for M3/M4 Max/Ultra
- [ ] Quantized inference (4-bit, 8-bit)
- [ ] Mobile deployment (iOS)

---

## Performance Expectations

### Training Time Estimates (d10, ~75M params)
- **8×H100 (PyTorch)**: ~30 minutes
- **1×M3 Max (MLX)**: ~4-8 hours
- **1×M2 Ultra (MLX)**: ~2-4 hours

### Memory Requirements
| Model | Params | PyTorch (GPU) | MLX (Unified) |
|-------|--------|---------------|---------------|
| d6 | ~20M | ~1GB | ~1GB |
| d10 | ~75M | ~3GB | ~3GB |
| d15 | ~200M | ~8GB | ~8GB |
| d20 | 561M | ~20GB | ~20GB |
| d26 | ~800M | ~32GB | ~32GB |

**Recommendation for Mac:**
- 32GB RAM: Train d10 (batch_size=1)
- 64GB RAM: Train d15 (batch_size=2)
- 96GB+ RAM: Train d20 (batch_size=4)

---

## Alternative Options (Not Chosen)

### Option A: Full Port (9-14 days)
- Complete MLX rewrite with distributed training
- **Rejected**: Too much effort for distributed, can add later

### Option B: Inference-Only Port (2-3 days)
- Port just model + inference, download pre-trained weights
- **Rejected**: Want full training capability for experimentation

---

## Resources & References

### MLX Documentation
- Main docs: https://ml-explore.github.io/mlx/
- Distributed: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- Examples: https://github.com/ml-explore/mlx-examples

### MLX Training Examples
- LLM training: https://github.com/ml-explore/mlx-examples/tree/main/llms
- LoRA fine-tuning: https://github.com/ml-explore/mlx-examples/tree/main/lora

### Community Projects
- MLX-LM: https://github.com/ml-explore/mlx-lm
- Distributed MLX: https://github.com/DaveAldon/Distributed-ML-with-MLX

---

## Notes

- Keep PyTorch version intact (on `master` branch)
- MLX port lives on `mlx-port` branch
- Can merge back to master once stable
- Document performance characteristics for Mac users
- Consider adding MLX as optional dependency (both frameworks supported)

---

**Last Updated:** October 15, 2025
**Status:** Planning Complete, Ready to Implement
