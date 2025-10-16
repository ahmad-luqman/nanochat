#!/bin/bash
# Exercise 1.1: Run a Pretrained Model
# Learn how to use a trained model for inference

echo "=================================================="
echo "Exercise 1.1: Run a Pretrained Model"
echo "=================================================="
echo ""

# Check if checkpoint exists
CHECKPOINT="checkpoints/d10_sft/best.npz"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "To complete this exercise:"
    echo "1. Train or download a model checkpoint"
    echo "2. Update CHECKPOINT path above"
    echo "3. Run: bash exercises/ex_1_1_run_model.sh"
    exit 1
fi

echo "✓ Checkpoint found: $CHECKPOINT"
echo ""

echo "=================================================="
echo "Basic Generation"
echo "=================================================="
echo ""

# Simple prompt
PROMPT="Once upon a time"
echo "Prompt: '$PROMPT'"
echo ""

python scripts/chat_cli_mlx.py \
    --checkpoint "$CHECKPOINT" \
    --mode plain \
    --prompt "$PROMPT" \
    --max-tokens 50 \
    --temperature 0.8 \
    --top-k 50

echo ""
echo "=================================================="
echo "Trying Different Prompts"
echo "=================================================="
echo ""

# Try different prompts
for prompt in \
    "The capital of France is" \
    "In machine learning, the term" \
    "Why is the sky"
do
    echo "Prompt: '$prompt'"
    python scripts/chat_cli_mlx.py \
        --checkpoint "$CHECKPOINT" \
        --mode plain \
        --prompt "$prompt" \
        --max-tokens 30 \
        --temperature 0.7
    echo ""
done

echo "=================================================="
echo "TODO: Experiments"
echo "=================================================="
echo ""
echo "1. Try different prompts"
echo "2. Change max-tokens (shorter/longer outputs)"
echo "3. Adjust temperature (0.1=deterministic, 1.5=creative)"
echo "4. Use --mode chat for interactive mode"
echo "5. Try different --top-k values"
echo ""
