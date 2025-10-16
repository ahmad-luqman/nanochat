#!/usr/bin/env python3
"""
Exercise 7.3: Model Evaluation - Assess Quality on Test Tasks

Learn how to:
- Create evaluation benchmarks
- Measure model performance
- Compare different models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.checkpoint_mlx import load_checkpoint
import mlx.core as mx
import mlx.nn as nn
import os
import json

print("=" * 60)
print("Model Evaluation: Testing on Benchmark Tasks")
print("=" * 60)

# Evaluation dataset - simple factual questions
eval_data = [
    {
        "prompt": "The capital of France is",
        "expected_token": "Paris",
        "category": "factual_geography"
    },
    {
        "prompt": "2 + 2 =",
        "expected_token": "4",
        "category": "math_simple"
    },
    {
        "prompt": "The largest planet in our solar system is",
        "expected_token": "Jupiter",
        "category": "factual_science"
    },
    {
        "prompt": "The opposite of hot is",
        "expected_token": "cold",
        "category": "semantic"
    },
    {
        "prompt": "In binary, the number 8 is written as",
        "expected_token": "1000",
        "category": "technical"
    },
]

def evaluate_model(checkpoint_path):
    """Evaluate model on benchmark tasks"""
    print(f"\nEvaluating: {checkpoint_path}")
    print(f"{'='*60}\n")

    if not os.path.exists(checkpoint_path + ".meta.json"):
        print(f"⚠️  Checkpoint metadata not found")
        return None

    # Load tokenizer
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    # Load model metadata
    meta_path = checkpoint_path + ".meta.json"
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    if 'model_config' not in metadata:
        print(f"⚠️  Model config not found in metadata")
        return None

    # Create model
    try:
        config = GPTConfig(**metadata['model_config'])
        model = GPT(config)
        model, _, _ = load_checkpoint(checkpoint_path, model=model)
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None

    # Evaluate
    results_by_category = {}
    all_results = []

    print(f"{'Prompt':<40s} | {'Expected':>15s} | {'Predicted':>15s} | {'Rank':>5s} | Status")
    print(f"{'-'*95}")

    for example in eval_data:
        prompt = example["prompt"]
        expected = example["expected_token"]
        category = example["category"]

        try:
            # Tokenize prompt
            tokens = tokenizer.encode(prompt)
            input_array = mx.array([tokens], dtype=mx.int32)

            # Get predictions
            logits = model(input_array)
            next_token_logits = logits[0, -1, :]

            # Get top 5 predictions
            top_5_ids = mx.argpartition(-next_token_logits, min(5, next_token_logits.size))[:5].tolist()
            top_5_tokens = [tokenizer.decode([tid]) for tid in top_5_ids]

            # Check if expected is in top 5
            expected_tokens = tokenizer.encode(expected)
            if expected_tokens:
                expected_id = expected_tokens[0]
                rank = None
                for i, tid in enumerate(top_5_ids):
                    if tid == expected_id:
                        rank = i + 1
                        break

                success = rank is not None
            else:
                success = False
                rank = None

            # Display
            pred_text = top_5_tokens[0] if top_5_tokens else "???"
            rank_str = f"#{rank}" if rank else "✗"
            status = "✓ CORRECT" if success else "✗ WRONG"

            prompt_display = prompt[:38] if len(prompt) <= 38 else prompt[:35] + "..."
            print(f"{prompt_display:<40s} | {expected:>15s} | {pred_text:>15s} | {rank_str:>5s} | {status}")

            # Track results
            if category not in results_by_category:
                results_by_category[category] = {"correct": 0, "total": 0}
            results_by_category[category]["total"] += 1
            if success:
                results_by_category[category]["correct"] += 1

            all_results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": top_5_tokens[0] if top_5_tokens else None,
                "rank": rank,
                "correct": success,
                "category": category,
            })

        except Exception as e:
            print(f"{prompt:<40s} | Error: {str(e)[:30]}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")

    total_correct = sum(r["correct"] for r in all_results)
    total_count = len(all_results)
    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    print(f"Overall Accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_count})")
    print(f"\nBy Category:")
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {category:<25s}: {acc*100:5.1f}% ({stats['correct']}/{stats['total']})")

    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "accuracy": overall_accuracy,
        "results": all_results,
        "by_category": results_by_category,
    }

# Evaluate available checkpoints
print("\n" + "="*60)
print("Evaluating Available Checkpoints")
print("="*60 + "\n")

checkpoints = [
    "checkpoints/d10_sft/best",
    "checkpoints/d10_pretrain_causal_fix/best",
]

eval_results = []
for ckpt in checkpoints:
    ckpt_path = ckpt
    result = evaluate_model(ckpt_path)
    if result:
        eval_results.append(result)

# Comparison
if len(eval_results) > 1:
    print(f"\n{'='*60}")
    print("Checkpoint Comparison")
    print(f"{'='*60}\n")

    print(f"{'Checkpoint':<25s} | {'Accuracy':<10s}")
    print(f"{'-'*40}")
    for r in eval_results:
        acc_str = f"{r['accuracy']*100:.1f}%"
        print(f"{r['checkpoint']:<25s} | {acc_str:<10s}")

print(f"\n{'='*60}")
print("Evaluation Methodology")
print(f"{'='*60}")

print("""
Types of Evaluation:

1. FACTUAL ACCURACY:
   - Test knowledge of facts
   - Strengths after pretraining
   - Scores: % correct predictions

2. REASONING:
   - Math problems, logic puzzles
   - Tests understanding
   - Harder to evaluate

3. INSTRUCTION FOLLOWING:
   - Does it follow directions?
   - Improved by SFT
   - Qualitative assessment

4. SEMANTIC UNDERSTANDING:
   - Opposite words, analogies
   - Tests embeddings
   - Can be quantified

5. CONSISTENCY:
   - Does it give same answer twice?
   - Important for production
   - Easy to measure

Benchmark selection:
- Should be representative
- Mix of difficulties
- Domain-specific if needed
- Should catch real issues
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Create more evaluation examples:
   - Add 20+ more examples
   - Different categories
   - Varying difficulty

2. Automated benchmarks:
   - MMLU (medical, history, etc.)
   - SQuAD (reading comprehension)
   - MATH (mathematical reasoning)

3. Human evaluation:
   - Ask people to rate responses
   - Score on multiple dimensions
   - Identify systematic issues

4. Compare models:
   - Pretrained vs. SFT
   - Different sizes
   - Track improvement

5. Error analysis:
   - What types of prompts fail?
   - Pattern in mistakes
   - How to improve?

6. Automated grading:
   - Exact match
   - Semantic similarity (embeddings)
   - Multiple correct answers
""")
