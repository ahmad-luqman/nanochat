#!/usr/bin/env python3
"""
Exercise 6.2: Masked Loss in SFT - Understanding Why We Train Only on Responses

Learn how to:
- Understand conversation formatting for SFT
- See how loss masking works
- Understand why this makes better assistants
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os

print("=" * 60)
print("Masked Loss in SFT: Train Only on Assistant Responses")
print("=" * 60)

# Load tokenizer
tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# Example conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}

print("\n1. Raw Conversation:")
print("-" * 60)
for msg in conversation["messages"]:
    print(f"  {msg['role'].upper()}: {msg['content']}")

# Render conversation
tokens, mask = tokenizer.render_conversation(conversation)

print("\n2. Tokenized Conversation:")
print("-" * 60)
print(f"  Total tokens: {len(tokens)}")
print(f"  Tokens: {tokens}")

print("\n3. Token-by-Token Analysis:")
print("-" * 60)
print(f"{'Pos':>4s} | {'Token ID':>8s} | {'Token Text':>25s} | {'Mask':>4s} | {'Status':>15s}")
print("-" * 70)

for i, (token_id, mask_val) in enumerate(zip(tokens, mask)):
    token_text = tokenizer.decode([token_id])
    # Limit display width
    token_display = token_text[:20] if len(token_text) <= 20 else token_text[:17] + "..."
    mask_indicator = "1" if mask_val == 1 else "0"
    status = "TRAIN (loss)" if mask_val == 1 else "SKIP (no loss)"
    print(f"{i:4d} | {token_id:8d} | {token_display:>25s} | {mask_indicator:>4s} | {status:>15s}")

# Count tokens by section
user_mask_sum = sum(1 for i, m in enumerate(mask) if m == 0 and i < len(mask) // 2)
assistant_mask_sum = sum(1 for m in mask if m == 1)
total_trained = sum(mask)

print(f"\n4. Summary:")
print("-" * 60)
print(f"  User tokens (mask=0): {user_mask_sum} tokens → Loss NOT computed")
print(f"  Assistant tokens (mask=1): {assistant_mask_sum} tokens → Loss computed")
print(f"  Total tokens trained: {total_trained}/{len(tokens)} ({total_trained*100/len(tokens):.1f}%)")

print(f"\n{'='*60}")
print("Why This Masking Matters")
print(f"{'='*60}")

print("""
The Key Insight:
We want the model to GENERATE good RESPONSES, not good PROMPTS!

WITHOUT masking (wrong):
  - Train on user questions: "What is 2+2?"
  - Model learns to predict: user_tokens → next_token
  - Problem: Model copies user's style, not generates responses

WITH masking (correct):
  - Skip user questions: mask = 0
  - Train on assistant answers: "2+2 equals 4."
  - Model learns to predict: response_tokens → next_token
  - Result: Model focuses on generating good responses

Example: Multi-turn conversation
  User: "Hello"
  Assistant: "Hi there!"
  User: "How are you?"
  Assistant: "I'm doing well, thanks for asking!"

WITHOUT masking:
  Losses on: all tokens (including user messages)
  Result: Model learns to continue conversations like the user

WITH masking:
  Losses only on: "Hi there!" and "I'm doing well, thanks for asking!"
  Result: Model learns to respond helpfully
""")

print(f"\n{'='*60}")
print("Multiple Conversations")
print(f"{'='*60}")

# Try another conversation
conversation2 = {
    "messages": [
        {"role": "user", "content": "Explain quantum computing"},
        {"role": "assistant", "content": "Quantum computing uses quantum bits..."}
    ]
}

tokens2, mask2 = tokenizer.render_conversation(conversation2)
assistant_count = sum(mask2)
total_count = len(tokens2)

print(f"\nConversation 2:")
for msg in conversation2["messages"]:
    print(f"  {msg['role'].upper()}: {msg['content'][:50]}...")

print(f"\n  Tokens: {total_count}")
print(f"  Trained on: {assistant_count}/{total_count} ({assistant_count*100/total_count:.1f}%)")
print(f"  Loss computed only on assistant response")

print(f"\n{'='*60}")
print("Implications for Training")
print(f"{'='*60}")

print("""
1. CONVERGENCE:
   - Model focuses gradient updates on assistant tokens
   - Faster learning of response patterns
   - Better final quality

2. DATA EFFICIENCY:
   - Every user-assistant pair counts as 1 training example
   - Masking makes each pair efficient
   - Less data needed vs. training on everything

3. AVOIDING CATASTROPHIC FORGETTING:
   - Model doesn't unlearn "how to continue" from pretraining
   - Only learns: how to respond to instructions
   - Preserves base knowledge

4. CONVERSATION STRUCTURE:
   - Tells model: "Here's a user query, here's a response"
   - Model learns mapping: query → good_response
   - Natural pairing for dialogue

5. MULTI-TURN CONSISTENCY:
   - Each assistant message gets full attention
   - Model learns context from user messages (softly)
   - But gradient only on assistant side
""")

print(f"\n{'='*60}")
print("Real-World SFT Data Format")
print(f"{'='*60}")

print("""
Typical SFT dataset structure:

[
  {
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Explain..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  ...
]

Masking automatically:
- Sets mask=0 for all user tokens
- Sets mask=1 for all assistant tokens
- Enables training focused on response generation

This is standard across:
- OpenAI's instruction-following models
- Meta's Llama fine-tuning
- Anthropic's RLHF pipeline
- Any serious SFT implementation
""")

print(f"\n{'='*60}")
print("TODO: Experiments")
print(f"{'='*60}")

print("""
1. Create a conversation with multiple turns
   - User: "..."
   - Assistant: "..."
   - User: "..."
   - Assistant: "..."
   - Verify mask pattern

2. Try different assistant response lengths
   - Short (1 token)
   - Long (50 tokens)
   - See how mask distribution changes

3. Understand the special tokens:
   - What are <|user_start|> and <|assistant_start|>?
   - How do they relate to masking?

4. Calculate: For a dataset with N examples
   - Average conversation length: L tokens
   - Typical user/assistant ratio: 40/60
   - How many tokens trained on vs. skipped?

5. Modify the mask manually
   - Set all mask=1 (wrong way)
   - Train and compare results
   - Understand the impact

6. Multi-turn conversations
   - Create 3-turn conversation
   - Verify each assistant response gets mask=1
""")
