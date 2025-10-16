#!/usr/bin/env python3
"""
Exercise 6.3: Understanding Hyperparameters
Learn what hyperparameters are and how they affect training

This exercise teaches:
- What hyperparameters control
- Why different values matter
- Tradeoffs in hyperparameter choices
- How to think about tuning them
"""

import sys
from pathlib import Path


def print_separator():
    print("=" * 80)


def print_hyperparameter_section(stage, params):
    """Print hyperparameters for a training stage"""
    print_separator()
    print(f"{stage} HYPERPARAMETERS")
    print_separator()
    print()

    # Extract why explanations
    why_dict = params.pop("why", {})

    for param, value in params.items():
        print(f"{param}: {value}")
        if param in why_dict:
            print(f"  → {why_dict[param]}")
        print()


def main():
    print_separator()
    print("Exercise 6.3: Understanding Hyperparameters")
    print_separator()
    print()
    print("Hyperparameters are the 'knobs' you turn to control training.")
    print("Let's understand what each one does!")
    print()

    # Part 1: What are hyperparameters?
    print_separator()
    print("Part 1: What Are Hyperparameters?")
    print_separator()
    print()
    print("Hyperparameters are settings you choose BEFORE training:")
    print("  - Model architecture (layers, heads, dimensions)")
    print("  - Training process (learning rate, batch size)")
    print("  - Data handling (sequence length, epochs)")
    print()
    print("They're different from model PARAMETERS (weights), which")
    print("are learned during training.")
    print()

    # Part 2: Pretraining hyperparameters
    pretrain_params = {
        "batch_size": 16,
        "learning_rate": 1e-3,
        "max_steps": 10000,
        "sequence_len": 1024,
        "warmup_steps": 500,
        "why": {
            "batch_size": "Larger = more stable gradients but uses more memory. 16 is a balance for MLX.",
            "learning_rate": "How big each weight update is. Too high = unstable, too low = slow learning.",
            "max_steps": "How many training iterations. More steps = better model but takes longer.",
            "sequence_len": "Maximum context length. Longer = more context but slower and more memory.",
            "warmup_steps": "Gradually increase LR at start to prevent instability in early training.",
        }
    }

    print_hyperparameter_section("PRETRAINING", pretrain_params)

    # Part 3: SFT hyperparameters
    sft_params = {
        "batch_size": 4,
        "learning_rate": 5e-5,
        "max_steps": 3000,
        "sequence_len": 512,
        "warmup_steps": 100,
        "why": {
            "batch_size": "Smaller than pretraining because: (1) sequences vary in length (padding waste), (2) less data.",
            "learning_rate": "MUCH lower than pretraining! We want to fine-tune, not drastically change the model.",
            "max_steps": "Fewer than pretraining. Fine-tuning is faster than learning from scratch.",
            "sequence_len": "Shorter than pretraining. Most instructions/responses are relatively short.",
            "warmup_steps": "Fewer than pretraining. Model is already trained, just adapting.",
        }
    }

    print_hyperparameter_section("SUPERVISED FINE-TUNING (SFT)", sft_params)

    # Part 4: Model architecture
    arch_params = {
        "n_layer": 10,
        "n_head": 8,
        "n_embd": 512,
        "vocab_size": 65536,
        "why": {
            "n_layer": "Number of transformer layers. More = more powerful but slower. 10 is medium-sized.",
            "n_head": "Number of attention heads. More = attend to more patterns. Usually 8-16.",
            "n_embd": "Embedding dimension. Larger = more capacity but more parameters. 512 is medium.",
            "vocab_size": "Determined by tokenizer (fixed). Larger vocab = fewer tokens per text.",
        }
    }

    print_hyperparameter_section("MODEL ARCHITECTURE", arch_params)

    # Part 5: Tradeoffs
    print_separator()
    print("Part 2: Understanding Tradeoffs")
    print_separator()
    print()

    tradeoffs = [
        ("Batch Size", [
            "Larger (32, 64, 128):",
            "  ✓ More stable gradients",
            "  ✓ Better GPU utilization",
            "  ✗ Uses more memory",
            "  ✗ Fewer updates per epoch",
            "",
            "Smaller (4, 8, 16):",
            "  ✓ Uses less memory",
            "  ✓ More updates per epoch",
            "  ✗ Noisier gradients",
            "  ✗ May need to adjust learning rate",
        ]),
        ("Learning Rate", [
            "Higher (1e-2, 1e-3):",
            "  ✓ Faster learning",
            "  ✗ Risk of instability",
            "  ✗ May overshoot optimal weights",
            "",
            "Lower (1e-5, 1e-6):",
            "  ✓ More stable",
            "  ✓ Fine-grained updates",
            "  ✗ Slower learning",
            "  ✗ May get stuck",
        ]),
        ("Model Size (n_layer, n_embd)", [
            "Larger (14 layers, 768 dim):",
            "  ✓ More capacity",
            "  ✓ Better performance",
            "  ✗ Slower training",
            "  ✗ Slower inference",
            "  ✗ More memory",
            "",
            "Smaller (6 layers, 384 dim):",
            "  ✓ Faster training",
            "  ✓ Faster inference",
            "  ✓ Less memory",
            "  ✗ Less capacity",
            "  ✗ May underfit",
        ]),
        ("Sequence Length", [
            "Longer (2048, 4096):",
            "  ✓ More context",
            "  ✓ Better long-range understanding",
            "  ✗ Much slower (quadratic in attention)",
            "  ✗ Much more memory",
            "",
            "Shorter (512, 1024):",
            "  ✓ Faster training",
            "  ✓ Less memory",
            "  ✗ Limited context",
            "  ✗ Can't handle long documents",
        ]),
    ]

    for param_name, details in tradeoffs:
        print(f"\n{param_name}:")
        print("-" * 80)
        for line in details:
            print(f"  {line}")

    print()

    # Part 6: Rules of thumb
    print_separator()
    print("Part 3: Rules of Thumb")
    print_separator()
    print()

    rules = [
        ("Starting a new project?", [
            "1. Start with a small model (d6 or d10)",
            "2. Use standard hyperparameters from similar work",
            "3. Train for a short time to verify setup works",
            "4. Then scale up model size and training time",
        ]),
        ("Fine-tuning a pretrained model?", [
            "1. Use LOWER learning rate than pretraining (10-100x lower)",
            "2. Use SMALLER batch size (data is usually limited)",
            "3. Train for FEWER steps (don't overfit!)",
            "4. Monitor validation loss carefully",
        ]),
        ("Getting poor results?", [
            "1. Check if loss is decreasing (if not → increase LR)",
            "2. Check if loss is diverging (if so → decrease LR)",
            "3. Try longer training (more steps)",
            "4. Try larger model (more capacity)",
            "5. Check your data quality!",
        ]),
        ("Out of memory?", [
            "1. Reduce batch_size",
            "2. Reduce sequence_len",
            "3. Reduce model size (n_layer, n_embd)",
            "4. Use gradient accumulation (simulate larger batches)",
        ]),
    ]

    for scenario, recommendations in rules:
        print(f"\n{scenario}")
        print("-" * 80)
        for rec in recommendations:
            print(f"  {rec}")

    print()

    # Part 7: Example configurations
    print_separator()
    print("Part 4: Example Configurations for Different Scenarios")
    print_separator()
    print()

    configs = [
        ("Quick Experiment (30 min on Mac)", {
            "model_size": "d2 or d6",
            "batch_size": 8,
            "max_steps": 1000,
            "sequence_len": 256,
            "learning_rate": "1e-3",
            "use_case": "Testing code, debugging, rapid iteration",
        }),
        ("Serious Training (few hours on Mac)", {
            "model_size": "d10",
            "batch_size": 16,
            "max_steps": 10000,
            "sequence_len": 1024,
            "learning_rate": "1e-3",
            "use_case": "Actual model for small projects",
        }),
        ("Production Model (days on GPU)", {
            "model_size": "d14 or d20",
            "batch_size": 32,
            "max_steps": 100000,
            "sequence_len": 2048,
            "learning_rate": "1e-3 with decay",
            "use_case": "High-quality model for real applications",
        }),
    ]

    for scenario, config in configs:
        print(f"\n{scenario}")
        print("-" * 80)
        for key, value in config.items():
            print(f"  {key:20s}: {value}")

    print()

    # Part 8: Interactive questions
    print_separator()
    print("Part 5: Test Your Understanding")
    print_separator()
    print()

    questions = [
        {
            "q": "You're training a model and the loss is 10.5, 10.3, 10.7, 9.2, 15.8, 200.3...",
            "a": "What's happening?",
            "answer": "Training is diverging! Learning rate is too high. Reduce it by 10x.",
        },
        {
            "q": "After 5000 steps, loss is 3.2. After 10000 steps, still 3.2.",
            "a": "What's happening?",
            "answer": "Model has converged. Either: (1) It's learned all it can, or (2) Learning rate is too low. Try training longer or increasing LR slightly.",
        },
        {
            "q": "Your validation loss is 2.5 but training loss is 0.1.",
            "a": "What's happening?",
            "answer": "Overfitting! Model memorized training data. Solutions: (1) More training data, (2) Smaller model, (3) Regularization, (4) Early stopping.",
        },
        {
            "q": "You increase batch_size from 16 to 64 and training gets worse.",
            "a": "Why?",
            "answer": "Larger batch size often needs larger learning rate. Try scaling LR proportionally (4x larger batch → maybe 2x larger LR).",
        },
    ]

    for i, qna in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"  {qna['q']}")
        print(f"  {qna['a']}")
        print(f"\n  Answer: {qna['answer']}")

    print()

    # Summary
    print_separator()
    print("Summary: Key Takeaways")
    print_separator()
    print()

    takeaways = [
        "✓ Hyperparameters control the training process",
        "✓ There's always a tradeoff (speed vs quality, memory vs performance)",
        "✓ Start with standard values, then adjust based on results",
        "✓ Learning rate is usually the most important hyperparameter",
        "✓ Bigger is not always better (risk of overfitting, memory issues)",
        "✓ Monitor your training! Loss plots tell you what to adjust",
        "✓ Different stages (pretrain vs SFT) need different settings",
    ]

    for takeaway in takeaways:
        print(f"  {takeaway}")

    print()
    print_separator()
    print("Try This Yourself")
    print_separator()
    print()
    print("Experiments to run:")
    print("  1. Train a d2 model with different learning rates (1e-2, 1e-3, 1e-4)")
    print("  2. Compare batch_size 4 vs 16 vs 32")
    print("  3. Try sequence_len 256 vs 1024 and measure speed difference")
    print("  4. Train same model for 1000 vs 5000 vs 10000 steps")
    print()
    print("For each experiment:")
    print("  - Track training loss over time")
    print("  - Measure training speed (tokens/sec)")
    print("  - Test final model quality")
    print("  - Understand the tradeoffs!")
    print()


if __name__ == "__main__":
    main()
