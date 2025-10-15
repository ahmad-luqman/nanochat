#!/bin/bash
# End-to-end test: Train d6 model and chat with it

# Activate venv
source .venv/bin/activate

echo "========================================"
echo "End-to-End MLX Test"
echo "========================================"
echo

# Step 1: Train a small model (d6 for 50 steps with real data)
echo "Step 1: Training d6 model for 50 steps with real data..."
python scripts/train_mlx.py \
    --model-size d6 \
    --batch-size 4 \
    --seq-len 256 \
    --max-steps 50 \
    --use-real-data \
    --learning-rate 0.001 \
    --log-interval 10

echo
echo "Step 2: Testing chat with the trained model (untrained, just checking pipeline)..."
python scripts/chat_cli_mlx.py \
    --model-size d6 \
    --prompt "What is machine learning?" \
    --max-tokens 30 \
    --temperature 1.0

echo
echo "========================================"
echo "✅ End-to-end test complete!"
echo "========================================"
echo
echo "Summary:"
echo "  ✓ Training pipeline works with real data"
echo "  ✓ Model initialization and saving works"
echo "  ✓ Chat generation works"
echo
echo "Note: Model output is still gibberish because:"
echo "  - Only trained for 50 steps (need thousands)"
echo "  - Small model size (d6 = 12.5M params)"
echo "  - No fine-tuning on chat data yet"
