#!/bin/bash
# Plan C - Full Training Runs
# This script contains commands for training d10, d14, and d20 models
# Run each command separately or let them run sequentially

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "========================================================================"
echo "Plan C - Full Training Pipeline"
echo "========================================================================"
echo ""

# ============================================================================
# Step 1: Download Full Dataset (Optional - if not already done)
# ============================================================================
echo "Step 1: Data Download"
echo "--------------------"
echo "This downloads all 1823 FineWebEdu shards (~100GB)"
echo "Skip this if you already have the data"
echo ""
read -p "Download data now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting data download (this will take 1-2 hours)..."
    python -m nanochat.dataset -n -1 -w 8 2>&1 | tee data_download.log
    echo "✓ Data download complete!"
    echo ""
fi

# ============================================================================
# Step 2: Train d10 Model (2-3 hours)
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 2: Training d10 Model (99M params)"
echo "========================================================================"
echo "Expected: 2-3 hours, ~13k tok/s, final loss ~5.5"
echo ""
read -p "Start d10 training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting d10 training..."
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
        --log-interval 100 \
        2>&1 | tee training_d10.log

    echo ""
    echo "✓ d10 training complete!"
    echo "  Checkpoint: checkpoints/d10_pretrain/best.npz"
    echo ""
fi

# ============================================================================
# Step 3: Train d14 Model (3-4 hours)
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 3: Training d14 Model (440M params)"
echo "========================================================================"
echo "Expected: 3-4 hours, ~10k tok/s, final loss ~5.2"
echo ""
read -p "Start d14 training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting d14 training..."
    python scripts/train_mlx.py \
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

    echo ""
    echo "✓ d14 training complete!"
    echo "  Checkpoint: checkpoints/d14_pretrain/best.npz"
    echo ""
fi

# ============================================================================
# Step 4: Train d20 Model [OPTIONAL] (4-5 hours)
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 4: Training d20 Model (1.03B params) [OPTIONAL]"
echo "========================================================================"
echo "Expected: 4-5 hours, ~7k tok/s, final loss ~4.8"
echo "Note: Requires ~80GB memory"
echo ""
read -p "Start d20 training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting d20 training..."
    python scripts/train_mlx.py \
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

    echo ""
    echo "✓ d20 training complete!"
    echo "  Checkpoint: checkpoints/d20_pretrain/best.npz"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Trained models:"
ls -lh checkpoints/*/best.npz 2>/dev/null || echo "  No checkpoints found"
echo ""
echo "Logs:"
ls -lh training_*.log 2>/dev/null || echo "  No logs found"
echo ""
echo "Next steps:"
echo "  1. Test models: python scripts/chat_cli_mlx.py --checkpoint checkpoints/d10_pretrain/best.npz"
echo "  2. Analyze metrics: cat checkpoints/d10_pretrain/metrics.json"
echo "  3. Compare models: Review training logs"
echo ""
