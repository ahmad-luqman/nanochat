# Plan C - Ready to Run Commands

All prep work is complete! You can now run these long training jobs.

## What's Ready

- ✅ d10, d14, d20 model configurations added
- ✅ All Plan B features integrated (validation, metrics, KV cache, LR scheduling)
- ✅ d10 model tested successfully (99M params, ~13k tok/s, 43% MFU)
- ✅ Training scripts ready

## Model Specifications

| Model | Params | Layers | Heads | Hidden | Seq Len | Vocab |
|-------|--------|--------|-------|--------|---------|-------|
| d10   | 99M    | 10     | 8     | 512    | 1024    | 65K   |
| d14   | 440M   | 14     | 12    | 768    | 1024    | 65K   |
| d20   | 1.03B  | 20     | 16    | 1024   | 1024    | 65K   |

---

## Quick Start (Copy-Paste Commands)

### Option 1: Run All Training Jobs Sequentially

```bash
# This will prompt you for each step
./train_plan_c.sh
```

### Option 2: Run Individual Commands

**Activate environment first:**
```bash
source .venv/bin/activate
```

---

## Step 1: Download Data (1-2 hours, ~100GB)

**Skip this if you already have data downloaded!**

```bash
python -m nanochat.dataset -n -1 -w 8 2>&1 | tee data_download.log
```

Check data:
```bash
ls -lh ~/.cache/nanochat/base_data/*.parquet | wc -l  # Should be 1823
```

---

## Step 2: Train d10 Model (2-3 hours)

**Copy-paste this entire command:**

```bash
source .venv/bin/activate && python scripts/train_mlx.py \
  --model-size d10 \
  --batch-size 32 \
  --seq-len 1024 \
  --max-steps 10000 \
  --use-real-data \
  --learning-rate 0.001 \
  --warmup-steps 1000 \
  --eval-interval 1000 \
  --val-batches 100 \
  --save-interval 1000 \
  --checkpoint-dir checkpoints/d10_pretrain \
  --log-interval 100 \
  2>&1 | tee training_d10.log
```

**Expected performance:**
- Time: 2-3 hours
- Throughput: ~13k tok/s
- MFU: ~43%
- Final val loss: ~5.5
- Final BPB: ~1.5

**Monitor training:**
```bash
# In another terminal
tail -f training_d10.log
```

---

## Step 3: Train d14 Model (3-4 hours)

**Copy-paste this entire command:**

```bash
source .venv/bin/activate && python scripts/train_mlx.py \
  --model-size d14 \
  --batch-size 24 \
  --seq-len 1024 \
  --max-steps 15000 \
  --use-real-data \
  --learning-rate 0.0008 \
  --warmup-steps 1500 \
  --eval-interval 1500 \
  --val-batches 100 \
  --save-interval 1500 \
  --checkpoint-dir checkpoints/d14_pretrain \
  --log-interval 100 \
  2>&1 | tee training_d14.log
```

**Expected performance:**
- Time: 3-4 hours
- Throughput: ~10k tok/s
- Final val loss: ~5.2
- Final BPB: ~1.4

---

## Step 4: Train d20 Model [OPTIONAL] (4-5 hours)

**⚠️ Requires ~80GB RAM**

```bash
source .venv/bin/activate && python scripts/train_mlx.py \
  --model-size d20 \
  --batch-size 16 \
  --seq-len 1024 \
  --max-steps 20000 \
  --use-real-data \
  --learning-rate 0.0006 \
  --warmup-steps 2000 \
  --eval-interval 2000 \
  --val-batches 100 \
  --save-interval 2000 \
  --checkpoint-dir checkpoints/d20_pretrain \
  --log-interval 100 \
  2>&1 | tee training_d20.log
```

**Expected performance:**
- Time: 4-5 hours
- Throughput: ~7k tok/s
- Final val loss: ~4.8
- Final BPB: ~1.3

---

## Running in Background (Recommended for overnight)

### Option A: Using `nohup`

```bash
nohup bash -c 'source .venv/bin/activate && python scripts/train_mlx.py \
  --model-size d10 \
  --batch-size 32 \
  --seq-len 1024 \
  --max-steps 10000 \
  --use-real-data \
  --learning-rate 0.001 \
  --warmup-steps 1000 \
  --eval-interval 1000 \
  --val-batches 100 \
  --save-interval 1000 \
  --checkpoint-dir checkpoints/d10_pretrain \
  --log-interval 100' > training_d10.log 2>&1 &
```

Check if running:
```bash
ps aux | grep train_mlx
```

Watch progress:
```bash
tail -f training_d10.log
```

### Option B: Using `screen` (better for long jobs)

```bash
# Start screen session
screen -S d10_training

# Inside screen, run training
source .venv/bin/activate
python scripts/train_mlx.py \
  --model-size d10 \
  --batch-size 32 \
  --seq-len 1024 \
  --max-steps 10000 \
  --use-real-data \
  --learning-rate 0.001 \
  --warmup-steps 1000 \
  --eval-interval 1000 \
  --val-batches 100 \
  --save-interval 1000 \
  --checkpoint-dir checkpoints/d10_pretrain \
  --log-interval 100

# Detach from screen: Press Ctrl+A, then D
# Re-attach later: screen -r d10_training
```

---

## After Training: Test Your Models

**Test d10:**
```bash
python scripts/chat_cli_mlx.py \
  --checkpoint checkpoints/d10_pretrain/best.npz \
  --prompt "What is machine learning?" \
  --max-tokens 100
```

**Analyze metrics:**
```bash
cat checkpoints/d10_pretrain/metrics.json | python -m json.tool
```

**Compare checkpoints:**
```bash
ls -lh checkpoints/d10_pretrain/
cat checkpoints/d10_pretrain/best.npz.meta.json
```

---

## What to Watch For

**Good signs:**
- Loss steadily decreasing
- Validation loss tracking training loss (not diverging)
- BPB going down toward 1.5 or lower
- No OOM errors
- Checkpoints saving every interval

**Warning signs:**
- Loss stuck or increasing
- Validation loss much higher than training (overfitting)
- OOM errors → reduce batch size
- Very slow (~<5k tok/s) → check system load

---

## Resume if Interrupted

If training crashes or you stop it, resume with:

```bash
python scripts/train_mlx.py \
  --resume-from checkpoints/d10_pretrain/step_8000.npz \
  --max-steps 10000
```

The model will:
- ✅ Load model weights
- ✅ Restore optimizer state
- ✅ Continue LR schedule
- ✅ Resume from exact step

---

## Recommended Workflow

**Tonight:**
1. Start data download if needed (1-2 hours)
2. Start d10 training before bed (2-3 hours)

**Tomorrow:**
1. Check d10 results
2. Start d14 training (3-4 hours)
3. Test d10 model while d14 trains

**Optional:**
- Run d20 if you want the largest model (~80GB RAM needed)

---

## Files You'll Get

For each model:
```
checkpoints/d10_pretrain/
├── best.npz              ← Best validation loss checkpoint
├── best.npz.meta.json    ← Metadata (step, loss, config)
├── best.npz.opt.npz      ← Optimizer state
├── step_1000.npz         ← Checkpoint at step 1000
├── step_2000.npz         ← Checkpoint at step 2000
├── ...
├── final_step_10000.npz  ← Final checkpoint
└── metrics.json          ← Training metrics log
```

---

## Help & Troubleshooting

**Out of memory:**
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--seq-len` (try 512 or 768)

**Training too slow:**
- Close other applications
- Check Activity Monitor for CPU/GPU usage
- Ensure you're using real data (faster than synthetic)

**Loss not decreasing:**
- Check learning rate isn't too high/low
- Ensure data is loading correctly
- Try more warmup steps

**Questions:**
- Check logs: `cat training_d10.log`
- Check metrics: `cat checkpoints/d10_pretrain/metrics.json`
- Come back to me with results!

---

## Summary

**Ready to run:**
1. Data download (optional, if needed)
2. d10 training (2-3 hours) ← Start with this!
3. d14 training (3-4 hours)
4. d20 training (4-5 hours, optional)

**Just copy-paste the commands above and let them run!**

When you're done, share:
- `training_d10.log`
- `checkpoints/d10_pretrain/metrics.json`
- Final metrics from the log

I'll analyze the results and create benchmark reports!
