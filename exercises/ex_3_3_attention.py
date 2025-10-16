#!/usr/bin/env python3
"""
Exercise 3.3: Understanding Attention (Conceptual)
Learn what attention does without complex math

This is a conceptual exercise to build intuition about:
- What is attention in Transformers
- Why we need multiple attention heads
- How attention enables context understanding
- The difference between self-attention and cross-attention
"""

import sys
from pathlib import Path


def print_separator():
    print("=" * 70)


def print_example(title, description, example):
    print(f"\n{title}")
    print("-" * 70)
    print(description)
    print()
    for line in example:
        print(f"  {line}")
    print()


def main():
    print_separator()
    print("Exercise 3.3: Understanding Attention (Conceptual)")
    print_separator()
    print()
    print("This exercise builds intuition about attention without complex math.")
    print("Read carefully and think about each example!")
    print()

    # Part 1: What is Attention?
    print_separator()
    print("Part 1: What is Attention?")
    print_separator()
    print()
    print("Imagine you're reading this sentence:")
    print("  'The ANIMAL didn't cross the STREET because IT was too tired.'")
    print()
    print("Question: What does 'IT' refer to?")
    print("  A) The animal")
    print("  B) The street")
    print()
    print("You know it's (A) the animal - but how?")
    print()
    print("You ATTENDED to relevant words:")
    print("  - 'IT' looks at → 'tired' (clue!)")
    print("  - 'tired' connects to → 'ANIMAL' (animals get tired)")
    print("  - Therefore: 'IT' = 'ANIMAL'")
    print()
    print("That's attention! Looking at relevant context to understand meaning.")
    print()

    # Part 2: Attention Mechanism
    print_separator()
    print("Part 2: How Attention Works (Simplified)")
    print_separator()
    print()
    print("For each word, the attention mechanism asks:")
    print()
    print("1. QUERY: 'What am I looking for?'")
    print("   - The current word's question")
    print("   - Example: 'IT' asks 'What noun am I referring to?'")
    print()
    print("2. KEY: 'What information is available?'")
    print("   - Each previous word's identity")
    print("   - Example: 'ANIMAL', 'STREET', 'tired' are all available")
    print()
    print("3. VALUE: 'What is that information?'")
    print("   - The actual meaning/representation of each word")
    print("   - Example: The semantic meaning of 'ANIMAL'")
    print()
    print("4. ATTENTION SCORES: similarity(Query, Keys)")
    print("   - How relevant is each word?")
    print("   - Example: 'ANIMAL' = high score, 'STREET' = low score")
    print()
    print("5. OUTPUT: weighted sum of Values")
    print("   - Combines information from relevant words")
    print("   - Example: 'IT' gets strong signal from 'ANIMAL'")
    print()

    # Part 3: Attention Example
    print_separator()
    print("Part 3: Attention Scores Example")
    print_separator()
    print()
    print("Sentence: 'The cat sat on the mat'")
    print("When processing 'mat', attention might look like:")
    print()

    attention_example = [
        "Position    Word     Attention Score   Why?",
        "--------    -----    ---------------   ----",
        "   0        'The'         0.05        Article (not very relevant)",
        "   1        'cat'         0.10        Subject (somewhat relevant)",
        "   2        'sat'         0.30        Verb (quite relevant - describes action)",
        "   3        'on'          0.40        Preposition (very relevant - shows location)",
        "   4        'the'         0.10        Article (not very relevant)",
        "   5        'mat'         0.05        Self (current word, less important)",
        "                        ------",
        "                         1.00        (scores sum to 1.0)",
    ]

    for line in attention_example:
        print(f"  {line}")

    print()
    print("The model attends most to 'on' and 'sat' to understand 'mat's role.")
    print()

    # Part 4: Multi-Head Attention
    print_separator()
    print("Part 4: Why Multiple Attention Heads?")
    print_separator()
    print()
    print("Different heads can focus on different aspects!")
    print()
    print("Sentence: 'The quick brown fox jumps over the lazy dog'")
    print()
    print("When processing 'fox':")
    print()

    heads_example = [
        "Head 1 (Grammar Head):",
        "  Attends to: 'The' (determiner), 'jumps' (verb)",
        "  Learning: Grammatical structure and relationships",
        "",
        "Head 2 (Semantic Head):",
        "  Attends to: 'quick', 'brown' (adjectives)",
        "  Learning: Descriptive properties of the fox",
        "",
        "Head 3 (Action Head):",
        "  Attends to: 'jumps', 'over' (action and direction)",
        "  Learning: What the fox is doing",
        "",
        "Head 4 (Subject-Object Head):",
        "  Attends to: 'dog' (object of the action)",
        "  Learning: Relationships between entities",
    ]

    for line in heads_example:
        print(f"  {line}")

    print()
    print("Multiple heads = multiple perspectives = richer understanding!")
    print()

    # Part 5: Causal Attention
    print_separator()
    print("Part 5: Causal Attention (Autoregressive)")
    print_separator()
    print()
    print("In language models, we use CAUSAL attention:")
    print("  - Each position can only attend to PREVIOUS positions")
    print("  - Cannot look ahead to future words")
    print("  - Why? Model must predict next word without seeing it!")
    print()
    print("Example: Predicting 'cat' in 'The cat sat'")
    print()

    causal_example = [
        "Position    Word     Can Attend To",
        "--------    -----    -------------",
        "   0        'The'    [nothing] (first word)",
        "   1        'cat'    ['The'] (can't see 'sat' yet!)",
        "   2        'sat'    ['The', 'cat']",
    ]

    for line in causal_example:
        print(f"  {line}")

    print()
    print("This is why it's called 'autoregressive' - each word depends only on past.")
    print()

    # Part 6: Attention Visualization
    print_separator()
    print("Part 6: Visualizing Attention")
    print_separator()
    print()
    print("If we could see attention, it would look like:")
    print()
    print("  Sentence: 'The cat sat on the mat'")
    print()
    print("  Attention from 'mat' to other words:")
    print()
    print("    The  cat  sat  on  the  mat")
    print("     ↓    ↓    ↓   ↓   ↓    ↓")
    print("    0.05 0.10 0.30 0.40 0.10 0.05")
    print("     │    │    │   │   │    │")
    print("     └────┴────┴───┴───┴────┘")
    print("              ↓")
    print("         'mat' representation")
    print()
    print("Thicker lines = stronger attention")
    print()

    # Part 7: Practical Implications
    print_separator()
    print("Part 7: Why Attention is Powerful")
    print_separator()
    print()

    benefits = [
        "1. Long-Range Dependencies:",
        "   - Can connect words far apart in text",
        "   - Example: 'Alice ... she' even with 100 words between",
        "",
        "2. Parallel Processing:",
        "   - All positions computed at once",
        "   - Unlike RNNs which process sequentially",
        "",
        "3. Interpretability:",
        "   - Can visualize what model attends to",
        "   - Helps debug and understand model behavior",
        "",
        "4. Flexibility:",
        "   - Different heads learn different patterns",
        "   - Adapts to various linguistic phenomena",
    ]

    for line in benefits:
        print(f"  {line}")

    print()

    # Part 8: Connection to Code
    print_separator()
    print("Part 8: Where to See This in Code")
    print_separator()
    print()
    print("In nanochat codebase:")
    print()
    print("  File: nanochat/gpt_mlx.py")
    print()
    print("  Key components:")
    print("    - MultiHeadAttention: The attention mechanism")
    print("    - n_head parameter: Number of attention heads (e.g., 8)")
    print("    - causal=True: Enables causal masking (can't look ahead)")
    print()
    print("  The attention block (simplified):")
    print()
    code_example = [
        "self.attention = nn.MultiHeadAttention(",
        "    dims=config.n_embd,      # Embedding dimension",
        "    num_heads=config.n_head,  # Number of heads (e.g., 8)",
        "    # causal=True means: can only attend to past",
        ")",
    ]
    for line in code_example:
        print(f"    {line}")
    print()

    # Summary
    print_separator()
    print("Summary: Key Takeaways")
    print_separator()
    print()

    takeaways = [
        "✓ Attention lets each word 'look at' relevant context",
        "✓ Query/Key/Value mechanism finds relevant information",
        "✓ Multiple heads = multiple perspectives on the text",
        "✓ Causal attention = can only look at past (for generation)",
        "✓ Attention scores show what the model focuses on",
        "✓ This is why Transformers work so well for language!",
    ]

    for takeaway in takeaways:
        print(f"  {takeaway}")

    print()
    print_separator()
    print("Think About These Questions")
    print_separator()
    print()

    questions = [
        "1. Why can't we use just one attention head?",
        "   (Hint: Different linguistic patterns)",
        "",
        "2. What would happen if we removed causal masking?",
        "   (Hint: Model could 'cheat' during training)",
        "",
        "3. How many attention heads should a model have?",
        "   (Hint: Tradeoff between diversity and computation)",
        "",
        "4. Can attention handle sentences of any length?",
        "   (Hint: Think about computation cost)",
        "",
        "5. What if two words are equally relevant?",
        "   (Hint: Both get attention, weighted sum)",
    ]

    for q in questions:
        print(f"  {q}")

    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("Now that you understand attention conceptually:")
    print("  1. Look at nanochat/gpt_mlx.py to see the implementation")
    print("  2. Try Exercise 3.2 to see attention in action")
    print("  3. Read the 'Attention is All You Need' paper")
    print("  4. Experiment with different numbers of heads in training")
    print()


if __name__ == "__main__":
    main()
