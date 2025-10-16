#!/usr/bin/env python3
"""
Exercise 7.4: Simple Chatbot - Build an Interactive Chat Application

Learn how to:
- Load a trained model
- Implement conversation loop
- Use KV cache for efficiency
- Create interactive application
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.kv_cache_mlx import KVCache
from nanochat.checkpoint_mlx import load_checkpoint
import mlx.core as mx
import os
import json

class SimpleChatbot:
    """A minimal chatbot implementation"""

    def __init__(self, checkpoint_path):
        """Initialize chatbot with a trained checkpoint"""
        print("Loading chatbot...", end=" ", flush=True)

        # Load tokenizer
        tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
        self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        # Load model
        meta_path = checkpoint_path + ".meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        config = GPTConfig(**metadata['model_config'])
        self.model = GPT(config)
        self.model, _, _ = load_checkpoint(checkpoint_path, model=self.model)

        # Setup KV cache for efficient generation
        self.kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.sequence_len * 10,  # Extra buffer
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer
        )

        # Special tokens
        self.bos = self.tokenizer.get_bos_token_id()
        try:
            self.user_start = self.tokenizer.encode_special("<|user_start|>")
            self.user_end = self.tokenizer.encode_special("<|user_end|>")
            self.assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
            self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        except:
            # Fallback if special tokens don't work
            self.user_start = [100274]  # Default placeholder
            self.user_end = [100275]
            self.assistant_start = [100276]
            self.assistant_end = [100277]

        # Conversation state
        self.conversation_tokens = [self.bos]
        print("✓")

    def chat(self, user_message, max_tokens=256, temperature=0.8, top_k=50):
        """Send a message and get a response"""
        # Add user message to conversation
        self.conversation_tokens.append(self.user_start)
        self.conversation_tokens.extend(self.tokenizer.encode(user_message))
        self.conversation_tokens.append(self.user_end)
        self.conversation_tokens.append(self.assistant_start)

        # Generate response
        response_tokens = []
        tokens_array = mx.array([self.conversation_tokens], dtype=mx.int32)
        logits = self.model(tokens_array, kv_cache=self.kv_cache)
        mx.eval(logits)

        for _ in range(max_tokens):
            next_token_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_logits = mx.topk(next_token_logits, k=min(top_k, next_token_logits.size))
                top_k_indices = mx.argtopk(next_token_logits, k=min(top_k, next_token_logits.size))
                next_token_logits = mx.full(next_token_logits.shape, -float('inf'))
                next_token_logits[top_k_indices] = top_k_logits

            next_token = mx.random.categorical(next_token_logits)
            next_token_id = int(next_token.item())

            # Stop if we hit assistant_end
            if next_token_id == self.assistant_end:
                break

            response_tokens.append(next_token_id)

            # Process next token
            next_input = mx.array([[next_token_id]], dtype=mx.int32)
            logits = self.model(next_input, kv_cache=self.kv_cache)
            mx.eval(logits)

        # Add to conversation
        response_tokens.append(self.assistant_end)
        self.conversation_tokens.extend(response_tokens)

        # Decode response
        response_text = self.tokenizer.decode(response_tokens[:-1])  # Exclude end token
        return response_text

    def reset(self):
        """Start a new conversation"""
        self.conversation_tokens = [self.bos]
        self.kv_cache.reset()

    def show_conversation(self):
        """Display the full conversation"""
        print(f"\n{'='*60}")
        print("Conversation History")
        print(f"{'='*60}\n")
        print(self.tokenizer.decode(self.conversation_tokens))

print("=" * 60)
print("Simple Chatbot - Interactive Chat")
print("=" * 60)

# Check if checkpoint exists
checkpoint_path = "checkpoints/d10_sft/best.npz"
if not os.path.exists(checkpoint_path):
    print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
    print("\nTo use this chatbot:")
    print("1. Train an SFT model: python scripts/sft_mlx.py ...")
    print("2. Or use an existing checkpoint")
    print("\nExample usage:")
    print("  bot = SimpleChatbot('checkpoints/d10_sft/best.npz')")
    print("  response = bot.chat('Hello, how are you?')")
    sys.exit(0)

try:
    # Initialize chatbot
    bot = SimpleChatbot(checkpoint_path)

    print("\n" + "=" * 60)
    print("Chatbot Ready!")
    print("=" * 60)
    print("""
Commands:
  - Type your message to chat
  - 'history' - Show conversation history
  - 'reset' - Start a new conversation
  - 'quit' - Exit

Generation parameters:
  - temperature: 0.8 (balance between coherence and creativity)
  - top_k: 50 (consider top 50 tokens)
""")

    # Interactive loop
    print("\nYou: ", end="", flush=True)
    for user_input in iter(lambda: input(), ""):
        user_input = user_input.strip()

        if not user_input:
            print("You: ", end="", flush=True)
            continue

        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break

        elif user_input.lower() == 'reset':
            bot.reset()
            print("(Conversation reset)\n")
            print("You: ", end="", flush=True)
            continue

        elif user_input.lower() == 'history':
            bot.show_conversation()
            print("\nYou: ", end="", flush=True)
            continue

        else:
            # Get response
            print("Bot: ", end="", flush=True)
            response = bot.chat(user_input, max_tokens=256, temperature=0.8, top_k=50)
            print(response)
            print(f"\nYou: ", end="", flush=True)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Chatbot Architecture")
print("=" * 60)

print("""
Components:

1. TOKENIZER:
   - Converts text ↔ tokens
   - Handles special tokens for roles
   - Manages vocabulary

2. MODEL:
   - GPT language model
   - Loaded from checkpoint
   - Generates next tokens

3. KV CACHE:
   - Stores Key/Value from attention
   - Speeds up generation 10-100x
   - Essential for interactive use

4. CONVERSATION STATE:
   - Maintains conversation history
   - Formats as tokens
   - Includes special role markers

5. GENERATION LOOP:
   - Iteratively generates tokens
   - Applies temperature/top-k
   - Stops at end token

Interaction Flow:
1. User types message
2. Encode to tokens
3. Add to conversation state
4. Generate response tokens one by one
5. Decode response
6. Display to user
7. Add to history for context

Key Features:
- Stateful (remembers context)
- Efficient (KV cache)
- Configurable (temp, top-k, max-tokens)
- Extensible (can add more features)
""")

print(f"\n{'='*60}")
print("TODO: Enhancements")
print(f"{'='*60}")

print("""
1. CONVERSATION MEMORY:
   - Trim old messages to stay in context window
   - Implement sliding window
   - Summarize context periodically

2. SYSTEM PROMPTS:
   - Add system message at start
   - Guide model behavior
   - Define persona/style

3. STREAMING RESPONSES:
   - Print tokens as generated
   - Better UX for long responses
   - Show generation in real-time

4. LOGGING:
   - Save conversations to file
   - Track metrics (avg latency, etc.)
   - Debug problematic inputs

5. TOOLS/ACTIONS:
   - Model can call functions
   - E.g., "search web for X"
   - Augment with external data

6. VOICE I/O:
   - Speech-to-text input
   - Text-to-speech output
   - Fully conversational

7. WEB UI:
   - Build a web interface
   - Deploy to server
   - Share with others
""")
