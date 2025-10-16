#!/usr/bin/env python3
"""
Exercise 2.3: Token Efficiency
Learn how different types of text tokenize

This exercise teaches:
- Why some texts tokenize more efficiently than others
- How tokenization affects model performance
- The relationship between characters and tokens
- Tokenizer bias towards English
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os


def analyze_text(tokenizer, text, category=""):
    """Analyze tokenization efficiency of a text"""
    tokens = tokenizer.encode(text)
    chars = len(text)
    token_count = len(tokens)
    ratio = chars / token_count if token_count > 0 else 0

    return {
        "text": text,
        "category": category,
        "chars": chars,
        "tokens": token_count,
        "ratio": ratio,
        "token_ids": tokens
    }


def main():
    print("=" * 70)
    print("Exercise 2.3: Token Efficiency")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    print("✓ Tokenizer loaded")
    print()

    # Experiment 1: Common vs rare words
    print("=" * 70)
    print("Experiment 1: Common Words vs Rare Words")
    print("=" * 70)
    print()

    word_tests = [
        ("the", "Very common"),
        ("hello", "Common"),
        ("computer", "Common technical"),
        ("antidisestablishmentarianism", "Rare long word"),
        ("pneumonoultramicroscopicsilicovolcanoconiosis", "Medical term"),
    ]

    print(f"{'Word':<50s} {'Category':<20s} {'Chars':<6s} {'Tokens':<8s} {'Ratio':<10s}")
    print("-" * 70)

    for word, category in word_tests:
        result = analyze_text(tokenizer, word, category)
        print(f"{result['text']:<50s} {result['category']:<20s} {result['chars']:<6d} {result['tokens']:<8d} {result['ratio']:<10.2f}")

    print()
    print("Notice: Common words are often single tokens, rare words are split")
    print()

    # Experiment 2: Different languages
    print("=" * 70)
    print("Experiment 2: Language Comparison")
    print("=" * 70)
    print()

    languages = [
        ("Hello, how are you?", "English"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("你好,你好吗?", "Chinese"),
        ("こんにちは、元気ですか?", "Japanese"),
        ("مرحبا، كيف حالك؟", "Arabic"),
        ("Привет, как дела?", "Russian"),
    ]

    print(f"{'Text':<40s} {'Language':<12s} {'Chars':<6s} {'Tokens':<8s} {'Efficiency':<12s}")
    print("-" * 70)

    for text, lang in languages:
        result = analyze_text(tokenizer, text, lang)
        efficiency = "Good" if result['ratio'] >= 3.0 else "Moderate" if result['ratio'] >= 2.0 else "Poor"
        print(f"{result['text']:<40s} {result['category']:<12s} {result['chars']:<6d} {result['tokens']:<8d} {efficiency:<12s}")

    print()
    print("Notice: Non-English languages often tokenize less efficiently")
    print("Why? Most BPE tokenizers are trained primarily on English text")
    print()

    # Experiment 3: Code vs natural language
    print("=" * 70)
    print("Experiment 3: Code vs Natural Language")
    print("=" * 70)
    print()

    code_vs_text = [
        ("This is a simple sentence.", "Natural language"),
        ("def hello(): print('Hi')", "Python code"),
        ("const x = 42;", "JavaScript"),
        ("<div class=\"container\">", "HTML"),
        ("SELECT * FROM users WHERE", "SQL"),
    ]

    print(f"{'Text':<35s} {'Type':<20s} {'Chars':<6s} {'Tokens':<8s} {'Ratio':<10s}")
    print("-" * 70)

    for text, text_type in code_vs_text:
        result = analyze_text(tokenizer, text, text_type)
        print(f"{result['text']:<35s} {result['category']:<20s} {result['chars']:<6d} {result['tokens']:<8d} {result['ratio']:<10.2f}")

    print()

    # Experiment 4: Numbers and special characters
    print("=" * 70)
    print("Experiment 4: Numbers and Special Characters")
    print("=" * 70)
    print()

    special_tests = [
        ("123", "Small numbers"),
        ("1234567890", "Long number"),
        ("3.14159265", "Decimal"),
        ("$100.00", "Currency"),
        ("user@email.com", "Email"),
        ("https://example.com", "URL"),
        ("🚀🌟💡❤️", "Emojis"),
        ("!!!", "Punctuation"),
    ]

    print(f"{'Text':<25s} {'Type':<20s} {'Chars':<6s} {'Tokens':<8s} {'Token IDs':<30s}")
    print("-" * 70)

    for text, text_type in special_tests:
        result = analyze_text(tokenizer, text, text_type)
        token_ids_str = str(result['token_ids'][:5])
        if len(result['token_ids']) > 5:
            token_ids_str += "..."
        print(f"{result['text']:<25s} {result['category']:<20s} {result['chars']:<6d} {result['tokens']:<8d} {token_ids_str:<30s}")

    print()

    # Experiment 5: Context window implications
    print("=" * 70)
    print("Experiment 5: Context Window Implications")
    print("=" * 70)
    print()

    # If model has 1024 token context window
    context_window = 1024

    print(f"Model context window: {context_window} tokens")
    print()

    example_texts = [
        ("English paragraph (4 chars/token)", 4.0),
        ("Chinese text (1.5 chars/token)", 1.5),
        ("Code (2.5 chars/token)", 2.5),
    ]

    print(f"{'Text Type':<35s} {'Chars per Token':<18s} {'Characters Fit':<20s}")
    print("-" * 70)

    for text_type, chars_per_token in example_texts:
        total_chars = int(context_window * chars_per_token)
        print(f"{text_type:<35s} {chars_per_token:<18.1f} {total_chars:<20,}")

    print()
    print("Notice: The same 1024-token window can hold:")
    print("  - ~4,000 characters of English text")
    print("  - ~1,500 characters of Chinese text")
    print("  - ~2,500 characters of code")
    print()

    # Summary
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("1. Tokenization Efficiency:")
    print("   - Common English words: Very efficient (high chars/token ratio)")
    print("   - Rare words: Less efficient (split into multiple tokens)")
    print("   - Non-English: Often less efficient (tokenizer bias)")
    print()
    print("2. Why This Matters:")
    print("   - Context window is measured in TOKENS, not characters")
    print("   - Same token limit = different character limits for different languages")
    print("   - Code can be less efficient than natural language")
    print()
    print("3. Impact on Model Performance:")
    print("   - More tokens = more computation required")
    print("   - Inefficient tokenization = less text fits in context")
    print("   - Models may perform worse on languages they didn't see much of")
    print()
    print("=" * 70)
    print("Try This Yourself (TODOs)")
    print("=" * 70)
    print()
    print("Modify this script and try:")
    print("1. Your own language (if not English)")
    print("2. Technical jargon from your field")
    print("3. Very long documents and track token usage")
    print("4. Mixed language text")
    print("5. Different programming languages")
    print()
    print("Questions to think about:")
    print("- How would you design a better multilingual tokenizer?")
    print("- What's the ideal vocabulary size? Tradeoffs?")
    print("- How does tokenization affect fairness across languages?")
    print()


if __name__ == "__main__":
    main()
