#!/bin/bash
# Exercise 6.1: Analyze Training Runs
# Look at training logs to understand the process

echo "=================================================="
echo "Exercise 6.1: Analyze Training Runs"
echo "=================================================="
echo ""

# Check if training log exists
if [ -f "sft_training.log" ]; then
    echo "Found training log: sft_training.log"
    echo ""

    echo "=================================================="
    echo "Training Progress (Loss Over Time)"
    echo "=================================================="
    echo ""

    # Show first 20 training steps
    echo "First 20 steps:"
    grep -E "Step|Loss|LR|loss" sft_training.log | head -20

    echo ""
    echo "=================================================="
    echo "Latest Training Steps"
    echo "=================================================="
    echo ""

    # Show last 20 steps
    echo "Last 20 steps:"
    grep -E "Step|Loss|loss" sft_training.log | tail -20

    echo ""
    echo "=================================================="
    echo "Training Statistics"
    echo "=================================================="
    echo ""

    # Count total steps
    total_steps=$(grep -c "Step" sft_training.log)
    echo "Total logged steps: $total_steps"

    # Find best loss
    best_loss=$(grep "loss" sft_training.log | grep -oE "loss.*[0-9]+\.[0-9]+" | sort -t: -k2 -n | head -1)
    echo "Best loss encountered: $best_loss"

    # Show learning rate changes
    echo ""
    echo "Learning rate schedule:"
    grep -E "LR|learning.rate" sft_training.log | head -10

else
    echo "⚠️  No training log found (sft_training.log)"
    echo ""
    echo "To create a training log:"
    echo "1. Run: python scripts/sft_mlx.py ... 2>&1 | tee sft_training.log"
    echo "2. Or check if training is running in background:"
    echo ""
fi

echo ""
echo "=================================================="
echo "What to Look For in Training"
echo "=================================================="
echo ""

cat << 'EOF'
Key Metrics:

1. LOSS:
   - Should decrease smoothly
   - May plateau after some steps
   - Plateauing = model has learned what it can

2. LEARNING RATE:
   - Usually starts small (warmup)
   - May stay constant or decay
   - Affects convergence speed

3. STEPS:
   - More steps = better model (usually)
   - With diminishing returns
   - Training time = (steps × time_per_step)

4. VALIDATION LOSS (if available):
   - Separates train/val performance
   - Big gap = overfitting
   - Same trend = good generalization

Pattern to Expect:
  Step 1-100:   Loss drops rapidly (warmup period)
  Step 100-500: Steady decrease
  Step 500+:    Slower decrease (diminishing returns)
  Final steps:  Plateau (convergence)

Red Flags:

  ✗ Loss increasing = LR too high
  ✗ Loss not changing = LR too low or stuck
  ✗ Loss NaN/Inf = Training diverged (fatal)
  ✗ Val loss >> train loss = Overfitting

Good Signs:

  ✓ Loss decreases every step (warmup)
  ✓ Smooth curve (no sudden jumps)
  ✓ Reaches plateau (convergence)
  ✓ Train ≈ Val loss (generalization)
EOF

echo ""
echo "=================================================="
echo "TODO: Experiments"
echo "=================================================="
echo ""
echo "1. Plot loss curve over steps"
echo "2. Calculate average loss reduction per 100 steps"
echo "3. Estimate when training plateaus"
echo "4. Compare different training runs"
echo "5. Find optimal number of training steps"
echo ""
