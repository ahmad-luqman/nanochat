# Plan C - Full Speedrun Training

**Status:** Planned
**Estimated Time:** 8-12 hours (mostly training time)
**Goal:** Train production-quality models and benchmark MLX vs PyTorch

## Prerequisites
- ✅ Plan A completed (core pipeline)
- ✅ Plan B completed (advanced features)
- ✅ All 1823 FineWebEdu shards downloaded (100GB)
- ⏳ M4 Max with 128GB RAM available for long runs

## Overview

Plan C involves full-scale training of nanochat models on Apple Silicon, comparable to the original PyTorch speedrun. We'll train progressively larger models and benchmark against the CUDA baseline.

---

## Phase C.1: Data Preparation (30 min)

### Tasks:
1. **Download full FineWebEdu dataset**
   - All 1823 shards (~100GB)
   - Use parallel download with resume capability

2. **Verify data integrity**
   - Check all parquet files are valid
   - Count total tokens available

3. **Create validation split**
   - Last shard is validation (already implemented)
   - Estimate tokens per epoch

### Commands:
```bash
# Download all shards (can take 1-2 hours depending on connection)
python -m nanochat.dataset -n -1 -w 8

# Verify data
ls -lh ~/.cache/nanochat/base_data/*.parquet | wc -l  # Should be 1823

# Estimate dataset size
python scripts/estimate_dataset_size.py
```

### Success criteria:
- [ ] All 1823 shards downloaded
- [ ] Data integrity verified
- [ ] ~100B tokens available for training

---

## Phase C.2: Model Training Runs (8-10 hours)

### Run 1: d10 Model (2-3 hours)
**Configuration:**
- Model size: d10 (36M params)
- Batch size: 32
- Sequence length: 1024
- Max steps: 10,000
- Learning rate: 0.001
- Warmup: 1,000 steps

**Command:**
```bash
python scripts/train_mlx.py \
    --model-size d10 \
    --batch-size 32 \
    --seq-len 1024 \
    --max-steps 10000 \
    --use-real-data \
    --learning-rate 0.001 \
    --warmup-steps 1000 \
    --save-interval 1000 \
    --checkpoint-dir checkpoints/d10_pretrain \
    --log-interval 100 \
    --use-muon
```

**Expected results:**
- Final loss: ~5.5
- BPB: ~1.5
- Training time: 2-3 hours
- Throughput: ~15k tok/s

---

### Run 2: d14 Model (3-4 hours)
**Configuration:**
- Model size: d14 (100M params)
- Batch size: 24
- Sequence length: 1024
- Max steps: 15,000
- Learning rate: 0.0008
- Warmup: 1,500 steps

**Command:**
```bash
python scripts/train_mlx.py \
    --model-size d14 \
    --batch-size 24 \
    --seq-len 1024 \
    --max-steps 15000 \
    --use-real-data \
    --learning-rate 0.0008 \
    --warmup-steps 1500 \
    --save-interval 1000 \
    --checkpoint-dir checkpoints/d14_pretrain \
    --log-interval 100 \
    --use-muon
```

**Expected results:**
- Final loss: ~5.2
- BPB: ~1.4
- Training time: 3-4 hours
- Throughput: ~12k tok/s

---

### Run 3: d20 Model (4-5 hours) [OPTIONAL]
**Configuration:**
- Model size: d20 (561M params)
- Batch size: 16
- Sequence length: 1024
- Max steps: 20,000
- Learning rate: 0.0006
- Warmup: 2,000 steps

**Command:**
```bash
python scripts/train_mlx.py \
    --model-size d20 \
    --batch-size 16 \
    --seq-len 1024 \
    --max-steps 20000 \
    --use-real-data \
    --learning-rate 0.0006 \
    --warmup-steps 2000 \
    --save-interval 2000 \
    --checkpoint-dir checkpoints/d20_pretrain \
    --log-interval 100 \
    --use-muon
```

**Expected results:**
- Final loss: ~4.8
- BPB: ~1.3
- Training time: 4-5 hours
- Throughput: ~8k tok/s

**Memory requirement:** ~80GB (should fit on M4 Max 128GB)

---

## Phase C.3: Supervised Fine-Tuning (SFT) (1-2 hours)

### Tasks:
1. **Prepare chat dataset**
   - Use OpenAssistant or similar chat dataset
   - Convert to nanochat conversation format
   - Tokenize with conversation markers

2. **Fine-tune best pretrained model**
   - Start from best d10 or d14 checkpoint
   - Train on chat data for 2-5K steps
   - Lower learning rate (0.0001)
   - Monitor validation loss carefully

### Command:
```bash
python scripts/train_sft_mlx.py \
    --checkpoint checkpoints/d14_pretrain/best.npz \
    --chat-data data/chat_dataset.jsonl \
    --batch-size 16 \
    --seq-len 2048 \
    --max-steps 5000 \
    --learning-rate 0.0001 \
    --warmup-steps 500 \
    --save-interval 500 \
    --checkpoint-dir checkpoints/d14_sft
```

### Success criteria:
- [ ] Model generates coherent chat responses
- [ ] Follows instruction format
- [ ] Loss converges without overfitting

---

## Phase C.4: Benchmarking & Evaluation (1 hour)

### Metrics to compare:

**1. Training Speed**
| Model | MLX (M4 Max) | PyTorch (8×H100) | Ratio |
|-------|--------------|------------------|-------|
| d10 | ~15k tok/s | ~120k tok/s | 8× slower |
| d14 | ~12k tok/s | ~100k tok/s | 8.3× slower |
| d20 | ~8k tok/s | ~60k tok/s | 7.5× slower |

**2. Final Loss**
| Model | MLX Loss | PyTorch Loss | Difference |
|-------|----------|--------------|------------|
| d10 | ? | 5.4 | ? |
| d14 | ? | 5.1 | ? |
| d20 | ? | 4.7 | ? |

**3. Generation Quality**
- Human evaluation on standard prompts
- Compare coherence, relevance, fluency
- Test multi-turn conversation ability

**4. Memory Usage**
| Model | MLX Peak Memory | PyTorch Peak Memory |
|-------|-----------------|---------------------|
| d10 | ~15GB | ~20GB (per GPU) |
| d14 | ~35GB | ~45GB |
| d20 | ~80GB | ~150GB |

**5. Cost Analysis**
- MLX: Free after M4 Max purchase ($3,500)
- PyTorch: ~$2-3/hour on cloud (8×H100)
- Breakeven point: ~1,200 hours of training

### Evaluation script:
```bash
# Run standard benchmark
python scripts/evaluate_mlx.py \
    --checkpoint checkpoints/d14_sft/best.npz \
    --benchmark hellaswag,mmlu,truthfulqa \
    --output results/d14_eval.json

# Compare with PyTorch baseline
python scripts/compare_results.py \
    --mlx results/d14_eval.json \
    --pytorch baseline/d14_pytorch.json
```

---

## Phase C.5: Documentation & Reporting (30 min)

### Deliverables:

1. **Training Report** (`docs/TRAINING_REPORT.md`)
   - Loss curves for all models
   - Training time and throughput
   - Hyperparameters used
   - Lessons learned

2. **Benchmark Results** (`docs/BENCHMARK_RESULTS.md`)
   - MLX vs PyTorch comparison
   - Memory usage analysis
   - Cost-benefit analysis
   - When to use MLX vs CUDA

3. **Deployment Guide** (`docs/DEPLOYMENT_MLX.md`)
   - How to serve models in production
   - Inference optimization tips
   - Scaling strategies
   - Integration with web frameworks

4. **Example Notebooks**
   - `notebooks/train_custom_model.ipynb`
   - `notebooks/fine_tune_chat.ipynb`
   - `notebooks/inference_demo.ipynb`

---

## Success Criteria

**Technical:**
- [ ] d10 model trained to loss < 5.5
- [ ] d14 model trained to loss < 5.2
- [ ] SFT model generates coherent chat responses
- [ ] All checkpoints saved and validated

**Performance:**
- [ ] Training speed within 80% of expectations
- [ ] Memory usage < 128GB for d20
- [ ] Generation quality comparable to PyTorch

**Documentation:**
- [ ] Complete training report with metrics
- [ ] Benchmark comparison with analysis
- [ ] Deployment guide with examples

---

## Timeline

**Day 1 (4 hours):**
- Morning: Data download and d10 training start
- Afternoon: Monitor d10, prepare d14

**Day 2 (4 hours):**
- Morning: d14 training
- Afternoon: SFT preparation and training

**Day 3 (2 hours):**
- Morning: Benchmarking and evaluation
- Afternoon: Documentation and reporting

**Total: 10 hours** (including monitoring time)

---

## Risks & Mitigations

**Risk 1: Training instability**
- Mitigation: Use gradient clipping, lower LR if loss spikes
- Fallback: Resume from last good checkpoint

**Risk 2: Out of memory**
- Mitigation: Reduce batch size, sequence length
- Fallback: Skip d20, focus on d14

**Risk 3: Data quality issues**
- Mitigation: Validate data loading, check for corruption
- Fallback: Use subset of high-quality shards

**Risk 4: Time overrun**
- Mitigation: Reduce max_steps for larger models
- Fallback: Skip d20, focus on producing one strong model

---

## Expected Outcomes

**Best case:**
- d20 model trained successfully (561M params)
- Loss within 5% of PyTorch baseline
- Chat capability demonstrated
- Complete benchmark report
- **Result:** Production-ready MLX training pipeline

**Realistic case:**
- d14 model trained successfully (100M params)
- Loss within 10% of PyTorch baseline
- Basic chat capability
- Partial benchmarks
- **Result:** Working MLX pipeline, room for optimization

**Worst case:**
- d10 model trained (36M params)
- Training completed but quality varies
- Basic generation works
- **Result:** Proof of concept, needs iteration

---

## Post-Plan C

After completing Plan C, you'll have:
1. **Trained models** ready for deployment
2. **Comprehensive benchmarks** for MLX on Apple Silicon
3. **Production pipeline** for training new models
4. **Knowledge base** for MLX optimization

Next steps:
- Scale to larger models (d26, d32)
- Explore quantization (4-bit, 8-bit)
- Implement RLHF pipeline
- Deploy to production environment
- Contribute findings back to MLX community
