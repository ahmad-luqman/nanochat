#!/bin/bash
# Exercise 7.1: Train a Model From Scratch (Mini Version)
# Train a small model end-to-end

echo "=================================================="
echo "Exercise 7.1: Train Model From Scratch"
echo "=================================================="
echo ""

echo "This will train a tiny d2 model on sample data"
echo "Expected time: ~5-10 minutes on Apple Silicon"
echo ""

# Create output directory
OUTPUT_DIR="exercises/my_first_model"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Stage 1: Pretraining"
echo "=================================================="
echo ""

echo "Starting pretraining (next token prediction)..."
echo "Training parameters:"
echo "  - Model: d2 (2 layers, 128 dims)"
echo "  - Batch size: 8"
echo "  - Max steps: 1000"
echo "  - Learning rate: 0.001"
echo ""

python scripts/train_mlx.py \
    --model-size d2 \
    --batch-size 8 \
    --max-steps 1000 \
    --learning-rate 0.001 \
    --warmup-steps 100 \
    --output-dir "$OUTPUT_DIR/pretrain" \
    --log-interval 50 \
    --save-interval 200

if [ ! -f "$OUTPUT_DIR/pretrain/best.npz" ]; then
    echo "❌ Pretraining failed - no checkpoint created"
    exit 1
fi

echo ""
echo "✓ Pretraining complete!"
echo "  Checkpoint: $OUTPUT_DIR/pretrain/best.npz"
echo ""

echo "=================================================="
echo "Stage 2: Test the Pretrained Model"
echo "=================================================="
echo ""

echo "Testing with different prompts:"
for prompt in \
    "Once upon a time" \
    "The future of AI" \
    "In the beginning"
do
    echo ""
    echo "Prompt: '$prompt'"
    python scripts/chat_cli_mlx.py \
        --checkpoint "$OUTPUT_DIR/pretrain/best.npz" \
        --mode plain \
        --prompt "$prompt" \
        --max-tokens 30 \
        --temperature 0.8
done

echo ""
echo "=================================================="
echo "What You Just Did"
echo "=================================================="
echo ""

cat << 'EOF'
🎉 You trained an LLM from scratch!

The model learned:
  1. Language patterns from data
  2. How to predict next tokens
  3. Vocabulary and embeddings
  4. Grammar and syntax

This is PRETRAINING - the foundation of modern LLMs.

The full pipeline:
  1. ✓ Pretraining (you just did this)
  2. ⬜ SFT (instruction tuning - optional)
  3. ⬜ RLHF (reinforcement learning - optional)

Next Steps:

Option 1: Fine-tune this model
  python scripts/sft_mlx.py \
    --base-checkpoint $OUTPUT_DIR/pretrain/best.npz \
    --output-dir $OUTPUT_DIR/sft

Option 2: Train a bigger model
  Change --model-size to d4 or d6
  Change --max-steps to 5000 or 10000

Option 3: Use real data
  Modify scripts to load real text data
  Then train to get better results
EOF

echo ""
echo "=================================================="
echo "Understanding the Training Process"
echo "=================================================="
echo ""

echo "During training, the model:"
echo "  1. Reads text data in batches"
echo "  2. Predicts next token for each position"
echo "  3. Computes loss (how wrong)"
echo "  4. Computes gradients (where to improve)"
echo "  5. Updates weights (learning!)"
echo "  6. Saves checkpoint at intervals"
echo ""

echo "Learning rate schedule:"
echo "  - Warmup: 0 → 0.001 (first 100 steps)"
echo "  - Stable: 0.001 (middle steps)"
echo "  - Optional: Decay (final steps)"
echo ""

echo "=================================================="
echo "TODO: Experiments"
echo "=================================================="
echo ""
echo "1. Check the loss curve:"
echo "   tail -50 exercises/my_first_model/pretrain/*.log"
echo ""
echo "2. Train longer (--max-steps 5000)"
echo "   Does loss keep decreasing?"
echo ""
echo "3. Bigger model (--model-size d4)"
echo "   How much slower?"
echo ""
echo "4. Different learning rate (--learning-rate 0.0005)"
echo "   Does it converge better?"
echo ""
echo "5. Use your own data:"
echo "   Modify train_mlx.py to load custom text"
echo ""
echo "6. SFT this model:"
echo "   python scripts/sft_mlx.py --base-checkpoint $OUTPUT_DIR/pretrain/best.npz"
echo ""
