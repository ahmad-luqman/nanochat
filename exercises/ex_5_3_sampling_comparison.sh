#!/bin/bash
# Exercise 5.3: Compare Different Sampling Strategies
# See how temperature and top-k affect output

echo "=================================================="
echo "Exercise 5.3: Sampling Strategies Comparison"
echo "=================================================="
echo ""

CHECKPOINT="checkpoints/d10_sft/best.npz"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "✓ Checkpoint found"
echo ""

PROMPT="What is machine learning?"

echo "=================================================="
echo "Sampling Strategy Comparison"
echo "=================================================="
echo ""
echo "Test Prompt: '$PROMPT'"
echo ""

echo "=================================================="
echo "1. DETERMINISTIC (temperature=0.1, top_k=1)"
echo "   Output: Same every time, boring"
echo "=================================================="
python scripts/chat_cli_mlx.py \
    --checkpoint "$CHECKPOINT" \
    --mode chat \
    --prompt "$PROMPT" \
    --max-tokens 50 \
    --temperature 0.1 \
    --top-k 1
echo ""

echo "=================================================="
echo "2. CONSERVATIVE (temperature=0.5, top_k=20)"
echo "   Output: Mostly top predictions, safe"
echo "=================================================="
python scripts/chat_cli_mlx.py \
    --checkpoint "$CHECKPOINT" \
    --mode chat \
    --prompt "$PROMPT" \
    --max-tokens 50 \
    --temperature 0.5 \
    --top-k 20
echo ""

echo "=================================================="
echo "3. BALANCED (temperature=0.8, top_k=50)"
echo "   Output: Natural, diverse but coherent"
echo "=================================================="
python scripts/chat_cli_mlx.py \
    --checkpoint "$CHECKPOINT" \
    --mode chat \
    --prompt "$PROMPT" \
    --max-tokens 50 \
    --temperature 0.8 \
    --top-k 50
echo ""

echo "=================================================="
echo "4. CREATIVE (temperature=1.5, top_k=100)"
echo "   Output: Diverse, sometimes odd"
echo "=================================================="
python scripts/chat_cli_mlx.py \
    --checkpoint "$CHECKPOINT" \
    --mode chat \
    --prompt "$PROMPT" \
    --max-tokens 50 \
    --temperature 1.5 \
    --top-k 100
echo ""

echo "=================================================="
echo "Key Observations"
echo "=================================================="
echo ""
echo "Temperature = 0.1:  Same output repeated multiple times"
echo "Temperature = 0.5:  Slightly varied but predictable"
echo "Temperature = 0.8:  Good balance (most natural)"
echo "Temperature = 1.5:  Very diverse, sometimes strange"
echo ""

echo "=================================================="
echo "Real-World Applications"
echo "=================================================="
echo ""
echo "Facts/Support:        use temperature=0.1-0.3"
echo "Chat/General:         use temperature=0.7-0.9"
echo "Brainstorming:        use temperature=1.2-1.5"
echo "Code generation:      use temperature=0.3-0.5"
echo ""

echo "=================================================="
echo "TODO: Experiments"
echo "=================================================="
echo ""
echo "1. Run each strategy 3 times - notice consistency"
echo "2. Try intermediate values (0.3, 0.6, 1.0, 1.2)"
echo "3. Test different prompts with same settings"
echo "4. Which temperature do you prefer?"
echo ""
