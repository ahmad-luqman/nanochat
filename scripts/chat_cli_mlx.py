#!/usr/bin/env python3
"""
Chat CLI for MLX models
Simple interactive chat interface for nanochat models trained with MLX

Usage:
    python scripts/chat_cli_mlx.py --checkpoint path/to/checkpoint.npz
    python scripts/chat_cli_mlx.py --model-size d6  # Use untrained model
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common_mlx import get_base_dir
from nanochat.checkpoint_mlx import load_checkpoint
from nanochat.kv_cache_mlx import KVCache
import os


def create_model(model_size='d6', vocab_size=None):
    """Create GPT model based on size"""
    configs = {
        'd2': GPTConfig(sequence_len=256, vocab_size=vocab_size or 65536, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
        'd6': GPTConfig(sequence_len=512, vocab_size=vocab_size or 65536, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
        'd10': GPTConfig(sequence_len=1024, vocab_size=vocab_size or 65536, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
        'd14': GPTConfig(sequence_len=1024, vocab_size=vocab_size or 65536, n_layer=14, n_head=12, n_kv_head=12, n_embd=768),
        'd20': GPTConfig(sequence_len=1024, vocab_size=vocab_size or 65536, n_layer=20, n_head=16, n_kv_head=16, n_embd=1024),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = GPT(config)
    model.init_weights()
    return model, config


def generate_tokens(model, tokenizer, prompt_tokens, max_tokens=256, temperature=0.8, top_k=50, repetition_penalty=1.2, kv_cache=None):
    """
    Generate tokens autoregressively using the model with KV cache for efficiency

    Args:
        model: GPT model
        tokenizer: Tokenizer
        prompt_tokens: List of prompt token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition)
        kv_cache: Optional KVCache object to use (if None, creates new one)

    Yields:
        token_id: Generated token ID
    """
    # Create KV cache if not provided
    if kv_cache is None:
        config = model.config
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,  # Use KV heads, not query heads
            seq_len=config.sequence_len * 10,  # Large buffer for long conversations
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer
        )

    # First pass: process the entire prompt to populate KV cache
    tokens = mx.array([prompt_tokens], dtype=mx.int32)
    logits = model(tokens, kv_cache=kv_cache)  # Process all prompt tokens at once
    mx.eval(logits)  # Materialize computation

    # Get logits for last position to start generation
    next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Track generated tokens for repetition penalty (only track recent ones)
    generated_tokens = []
    repetition_window = 50  # Only penalize tokens from last 50 generations

    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            # Only penalize recent tokens (last repetition_window tokens)
            recent_tokens = generated_tokens[-repetition_window:]
            logits_array = next_token_logits.tolist()

            for token_id in set(recent_tokens):
                # Apply penalty: divide logit by penalty if positive, multiply if negative
                if logits_array[token_id] > 0:
                    logits_array[token_id] = logits_array[token_id] / repetition_penalty
                else:
                    logits_array[token_id] = logits_array[token_id] * repetition_penalty

            next_token_logits = mx.array(logits_array)

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            k = min(top_k, next_token_logits.shape[-1])
            top_vals = mx.topk(next_token_logits, k)
            threshold = mx.min(top_vals)
            next_token_logits = mx.where(
                next_token_logits >= threshold,
                next_token_logits,
                mx.array(-float('inf'))
            )

        # Sample from distribution (categorical expects logits, not probs)
        next_token = mx.random.categorical(next_token_logits)

        # Convert to Python int
        next_token_id = int(next_token.item())

        # Track this token for repetition penalty
        generated_tokens.append(next_token_id)

        # Yield the token
        yield next_token_id

        # For next iteration: only process the single new token (KV cache magic!)
        # This is MUCH faster than reprocessing the entire sequence
        next_tokens = mx.array([[next_token_id]], dtype=mx.int32)
        logits = model(next_tokens, kv_cache=kv_cache)
        mx.eval(logits)
        next_token_logits = logits[0, -1, :]


def main():
    parser = argparse.ArgumentParser(description='Chat with MLX nanochat model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (.npz)')
    parser.add_argument('--model-size', type=str, default='d6', choices=['d2', 'd6', 'd10', 'd14', 'd20'],
                       help='Model size (if not loading checkpoint)')
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('-k', '--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum tokens to generate')
    parser.add_argument('-p', '--prompt', type=str, default='',
                       help='Single prompt mode (non-interactive)')
    parser.add_argument('--mode', type=str, default='plain', choices=['plain', 'chat'],
                       help='Inference mode: plain (base model) or chat (SFT model)')

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Tokenizer loaded (vocab size: {vocab_size:,})")

    # Load or create model
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        # Load metadata to get config
        import json
        meta_path = args.checkpoint + ".meta.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model from config
        from nanochat.gpt_mlx import GPTConfig
        config_dict = metadata['model_config']
        config = GPTConfig(**config_dict)
        model = GPT(config)

        # Load weights into model
        model, _, _ = load_checkpoint(args.checkpoint, model=model)
        print(f"✓ Model loaded: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    else:
        print(f"\nCreating {args.model_size} model (untrained)...")
        model, config = create_model(args.model_size, vocab_size=vocab_size)
        print(f"✓ Model created: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
        print(f"⚠️  Using untrained model - responses will be random!")

    # Get special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    print("\n" + "=" * 60)
    print(f"NanoChat MLX - {args.mode.upper()} Mode")
    print("=" * 60)
    print("Commands:")
    print("  'quit' or 'exit' - End conversation")
    print("  'clear' - Start new conversation")
    print("=" * 60)
    if args.mode == 'plain':
        print("📝 Plain text mode (for base pretrained models)")
    else:
        print("💬 Chat mode (for SFT models)")
    print("✨ Using KV cache for fast generation!")
    print("=" * 60)

    # Initialize conversation and KV cache
    conversation_tokens = [bos]

    # Create KV cache for multi-turn conversations
    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len * 10,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer
    )

    # Single prompt mode
    if args.prompt:
        user_input = args.prompt
        print(f"\n{'User' if args.mode == 'chat' else 'Prompt'}: {user_input}")

        if args.mode == 'chat':
            # Chat mode: use special tokens
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(user_input))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
        else:
            # Plain mode: just encode the text directly
            conversation_tokens = tokenizer.encode(user_input)

        print(f"\n{'Assistant' if args.mode == 'chat' else 'Generated'}: ", end="", flush=True)
        response_tokens = []
        for token_id in generate_tokens(
            model, tokenizer, conversation_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            kv_cache=kv_cache
        ):
            response_tokens.append(token_id)
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)

            # Stop if we hit assistant_end (only in chat mode)
            if args.mode == 'chat' and token_id == assistant_end:
                break

        print("\n")
        return

    # Interactive mode
    while True:
        try:
            prompt_label = "User" if args.mode == 'chat' else "Prompt"
            user_input = input(f"\n{prompt_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_tokens = [bos] if args.mode == 'chat' else []
            kv_cache.reset()  # Reset the KV cache
            print("Conversation cleared (KV cache reset).")
            continue

        if not user_input:
            continue

        if args.mode == 'chat':
            # Chat mode: use special tokens
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(user_input))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
        else:
            # Plain mode: just encode the text directly
            # For interactive, append to existing tokens to maintain context
            if conversation_tokens:
                conversation_tokens.extend(tokenizer.encode(" " + user_input))
            else:
                conversation_tokens = tokenizer.encode(user_input)

        response_label = "Assistant" if args.mode == 'chat' else "Generated"
        print(f"\n{response_label}: ", end="", flush=True)
        response_tokens = []
        for token_id in generate_tokens(
            model, tokenizer, conversation_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            kv_cache=kv_cache
        ):
            response_tokens.append(token_id)
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)

            # Stop if we hit assistant_end (only in chat mode)
            if args.mode == 'chat' and token_id == assistant_end:
                break

        # Ensure assistant_end is at the end (only in chat mode)
        if args.mode == 'chat':
            if response_tokens and response_tokens[-1] != assistant_end:
                response_tokens.append(assistant_end)

        conversation_tokens.extend(response_tokens)
        print()


if __name__ == "__main__":
    main()
