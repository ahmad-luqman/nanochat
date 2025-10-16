#!/usr/bin/env python3
"""
Exercise 2.2: Special Tokens
Learn about special tokens and conversation formatting

This exercise teaches:
- What special tokens are and why they matter
- How conversations are structured
- How the loss mask works (what the model learns from)
- Why SFT models need special formatting
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os


def main():
    print("=" * 70)
    print("Exercise 2.2: Special Tokens and Conversation Formatting")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    print("✓ Tokenizer loaded")
    print()

    # Experiment 1: Special tokens
    print("=" * 70)
    print("Experiment 1: Special Tokens")
    print("=" * 70)
    print("\nSpecial tokens mark different parts of a conversation:")
    print()

    special_tokens = {
        "BOS (Beginning of Sequence)": tokenizer.get_bos_token_id(),
        "User Start": tokenizer.encode_special("<|user_start|>"),
        "User End": tokenizer.encode_special("<|user_end|>"),
        "Assistant Start": tokenizer.encode_special("<|assistant_start|>"),
        "Assistant End": tokenizer.encode_special("<|assistant_end|>"),
    }

    for name, token_id in special_tokens.items():
        print(f"  {name:30s}: Token ID = {token_id}")

    print()
    print("Why special tokens matter:")
    print("  - They tell the model: 'this is where the user speaks'")
    print("  - And: 'this is where the assistant responds'")
    print("  - The model learns these boundaries during training")
    print()

    # Experiment 2: Format a conversation
    print("=" * 70)
    print("Experiment 2: Conversation Formatting")
    print("=" * 70)
    print()

    conversation = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]
    }

    print("Input conversation:")
    for msg in conversation["messages"]:
        print(f"  {msg['role']:10s}: {msg['content']}")
    print()

    # Render conversation
    tokens, mask = tokenizer.render_conversation(conversation)

    print("Tokenized conversation:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Token IDs: {tokens}")
    print()

    # Experiment 3: Understanding the loss mask
    print("=" * 70)
    print("Experiment 3: Understanding the Loss Mask")
    print("=" * 70)
    print()
    print("The mask tells the model which tokens to learn from:")
    print("  mask=0: SKIP (don't compute loss)")
    print("  mask=1: LEARN (compute loss, train on this)")
    print()

    print(f"{'Position':<10s} {'Token ID':<10s} {'Token Text':<25s} {'Mask':<6s} {'Learn?':<10s}")
    print("-" * 70)

    for i, (token_id, mask_val) in enumerate(zip(tokens, mask)):
        token_text = tokenizer.decode([token_id])
        # Make special characters visible
        display_text = token_text.replace(" ", "␣").replace("\n", "↵")
        display_text = display_text[:22] + "..." if len(display_text) > 22 else display_text

        learn_status = "✓ TRAIN" if mask_val == 1 else "✗ SKIP"

        print(f"{i:<10d} {token_id:<10d} {display_text:<25s} {mask_val:<6d} {learn_status:<10s}")

    print()
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("1. BOS token marks the start of the conversation")
    print("2. User messages are wrapped in <|user_start|> ... <|user_end|>")
    print("3. Assistant messages are wrapped in <|assistant_start|> ... <|assistant_end|>")
    print("4. The model only learns from assistant tokens (mask=1)")
    print("5. User prompt tokens are skipped (mask=0) - we don't train the model")
    print("   to generate user messages!")
    print()

    # Experiment 4: Multi-turn conversation
    print("=" * 70)
    print("Experiment 4: Multi-Turn Conversation")
    print("=" * 70)
    print()

    multi_turn = {
        "messages": [
            {"role": "user", "content": "Hi! What's your name?"},
            {"role": "assistant", "content": "I'm an AI assistant. How can I help you today?"},
            {"role": "user", "content": "Can you explain what you are?"},
            {"role": "assistant", "content": "I'm a language model trained to be helpful, harmless, and honest."}
        ]
    }

    print("Multi-turn conversation:")
    for i, msg in enumerate(multi_turn["messages"], 1):
        print(f"  Turn {i} ({msg['role']:9s}): {msg['content']}")
    print()

    tokens, mask = tokenizer.render_conversation(multi_turn)
    print(f"Tokenized:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Tokens with mask=1 (training): {sum(mask)}")
    print(f"  Tokens with mask=0 (skipped): {len(mask) - sum(mask)}")
    print()

    # Show just assistant parts
    print("Assistant responses only (what the model learns to generate):")
    assistant_mode = False
    assistant_tokens = []

    for token_id, mask_val in zip(tokens, mask):
        token_text = tokenizer.decode([token_id])

        if "<|assistant_start|>" in token_text:
            assistant_mode = True
            continue
        elif "<|assistant_end|>" in token_text:
            if assistant_tokens:
                response = tokenizer.decode(assistant_tokens)
                print(f"  → '{response}'")
            assistant_tokens = []
            assistant_mode = False
            continue

        if assistant_mode and mask_val == 1:
            assistant_tokens.append(token_id)

    print()
    print("=" * 70)
    print("Try This Yourself (TODOs)")
    print("=" * 70)
    print()
    print("Modify this script and try:")
    print("1. Create your own conversation (3+ turns)")
    print("2. Add a conversation with code in it")
    print("3. Try a conversation with multiple questions in one turn")
    print("4. Create a conversation in a different language")
    print()
    print("Questions to think about:")
    print("- Why do we skip user tokens in training?")
    print("- What happens if we trained on user tokens too?")
    print("- How does the model know when to stop generating?")
    print("- Why do we need special tokens instead of just using text?")
    print()


if __name__ == "__main__":
    main()
