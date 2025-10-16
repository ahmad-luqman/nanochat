#!/usr/bin/env python3
"""
Exercise 2.1: Tokenizer Basics
Learn how text is converted to numbers and back

This exercise teaches:
- How to encode text into token IDs
- How to decode token IDs back to text
- How to see individual tokens
- Why tokenization is needed for LLMs
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import nanochat modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os


def main():
    print("=" * 70)
    print("Exercise 2.1: Tokenizer Basics")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Tokenizer loaded (vocabulary size: {vocab_size:,} tokens)")
    print()

    # Experiment 1: Encode simple text
    print("=" * 70)
    print("Experiment 1: Encoding Text")
    print("=" * 70)
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    print(f"Text: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print()

    # Experiment 2: Decode back
    print("=" * 70)
    print("Experiment 2: Decoding Tokens")
    print("=" * 70)
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    print(f"Match original? {decoded == text}")
    print()

    # Experiment 3: See individual tokens
    print("=" * 70)
    print("Experiment 3: Individual Tokens")
    print("=" * 70)
    print(f"Breaking down: '{text}'")
    print()
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        # Show special characters more clearly
        display_text = token_text.replace(" ", "␣").replace("\n", "↵")
        print(f"  Position {i}: Token ID {token_id:6d} → '{display_text}'")
    print()

    # Experiment 4: Different types of text
    print("=" * 70)
    print("Experiment 4: How Different Texts Are Tokenized")
    print("=" * 70)

    test_texts = [
        "Hello",  # Simple word
        "Supercalifragilisticexpialidocious",  # Very long word
        "123456789",  # Numbers
        "The quick brown fox jumps over the lazy dog",  # Sentence
        "print('Hello, World!')",  # Code
        "🚀🌟💡",  # Emojis
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        chars = len(text)
        token_count = len(tokens)
        ratio = chars / token_count if token_count > 0 else 0

        print(f"\nText: '{text}'")
        print(f"  Characters: {chars}, Tokens: {token_count}, Ratio: {ratio:.2f} chars/token")
        print(f"  Token IDs: {tokens}")

        # Show breakdown for short texts
        if token_count <= 10:
            print(f"  Breakdown:")
            for token_id in tokens:
                token_text = tokenizer.decode([token_id])
                display_text = token_text.replace(" ", "␣").replace("\n", "↵")
                print(f"    {token_id:6d} → '{display_text}'")

    print()
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("1. Tokenization converts text → numbers (computers understand numbers)")
    print("2. Each token is a piece of text (word, part of word, or character)")
    print("3. Common words are usually single tokens")
    print("4. Rare or long words may be split into multiple tokens")
    print("5. Numbers and special characters have their own tokenization patterns")
    print()
    print("=" * 70)
    print("Try This Yourself (TODOs)")
    print("=" * 70)
    print()
    print("Modify this script and try:")
    print("1. Different languages (Spanish, French, Chinese)")
    print("2. Very long sentences")
    print("3. Mix of code and natural language")
    print("4. Special characters and symbols")
    print("5. Compare tokenization efficiency across different types of text")
    print()
    print("Questions to think about:")
    print("- Why do some words tokenize more efficiently than others?")
    print("- How does the tokenizer handle unknown words?")
    print("- What happens with text the tokenizer hasn't seen before?")
    print()


if __name__ == "__main__":
    main()
