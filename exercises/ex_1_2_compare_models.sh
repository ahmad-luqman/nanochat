#!/bin/bash
# Exercise 1.2: Compare Base vs SFT Models
# See the difference between pretrained and instruction-tuned

echo "=================================================="
echo "Exercise 1.2: Compare Base vs SFT Models"
echo "=================================================="
echo ""

# Check checkpoints
BASE_CHECKPOINT="checkpoints/d10_pretrain_causal_fix/best.npz"
SFT_CHECKPOINT="checkpoints/d10_sft/best.npz"

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "❌ Base checkpoint not found: $BASE_CHECKPOINT"
    exit 1
fi

if [ ! -f "$SFT_CHECKPOINT" ]; then
    echo "❌ SFT checkpoint not found: $SFT_CHECKPOINT"
    exit 1
fi

echo "✓ Both checkpoints found"
echo ""

# Test prompts
PROMPTS=(
    "What is the capital of France?"
    "Explain quantum computing"
    "How do I make pizza?"
)

echo "=================================================="
echo "Comparison Test Prompts"
echo "=================================================="
echo ""

for prompt in "${PROMPTS[@]}"; do
    echo "=================================================="
    echo "Prompt: '$prompt'"
    echo "=================================================="
    echo ""

    echo "BASE MODEL (pretrained):"
    echo "─────────────────────────"
    python scripts/chat_cli_mlx.py \
        --checkpoint "$BASE_CHECKPOINT" \
        --mode plain \
        --prompt "$prompt" \
        --max-tokens 40 \
        --temperature 0.8
    echo ""

    echo "SFT MODEL (instruction-tuned):"
    echo "───────────────────────────────"
    python scripts/chat_cli_mlx.py \
        --checkpoint "$SFT_CHECKPOINT" \
        --mode chat \
        --prompt "$prompt" \
        --max-tokens 40 \
        --temperature 0.8
    echo ""
    echo ""
done

echo "=================================================="
echo "Key Differences to Observe"
echo "=================================================="
echo ""

echo "BASE MODEL:"
echo "  ✗ Ignores question format"
echo "  ✗ Tries to continue the text"
echo "  ✓ Good language patterns from pretraining"
echo ""

echo "SFT MODEL:"
echo "  ✓ Understands it's a question"
echo "  ✓ Tries to answer helpfully"
echo "  ✓ Follows instruction format"
echo ""

echo "Why the difference?"
echo "  Base: Only saw 'next token prediction' during training"
echo "  SFT:  Specifically trained on Q&A pairs with masking"
echo "  Result: SFT is much better at following instructions!"
echo ""

echo "=================================================="
echo "TODO: Experiments"
echo "=================================================="
echo ""
echo "1. Try more prompts"
echo "2. Use same temperature for fair comparison"
echo "3. Notice which one sounds more helpful"
echo "4. Try rephrasing the same question different ways"
echo "5. Test on facts, reasoning, creative writing"
echo ""
