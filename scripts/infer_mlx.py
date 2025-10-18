#!/usr/bin/env python3
"""
Simple inference script for nanochat MLX models
Generates text from a prompt using trained model
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache


def create_model(model_size='d6'):
    """Create GPT model based on size"""
    configs = {
        'd2': GPTConfig(sequence_len=256, vocab_size=2048, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
        'd6': GPTConfig(sequence_len=512, vocab_size=4096, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
        'd10': GPTConfig(sequence_len=1024, vocab_size=8192, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = GPT(config)
    model.init_weights()
    return model, config


def generate_text(
    model,
    config,
    prompt_tokens,
    max_tokens=100,
    temperature=1.0,
    top_k=None,
    use_kv_cache=True,
    verbose=True
):
    """
    Generate text from a prompt using the model.

    Args:
        model: GPT model
        config: Model config
        prompt_tokens: List of token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k sampling (None = sample from full distribution)
        use_kv_cache: Whether to use KV cache for efficiency
        verbose: Print generation statistics

    Returns:
        generated_tokens: List of generated token IDs (excluding prompt)
    """
    if verbose:
        print(f"Prompt length: {len(prompt_tokens)} tokens")
        print(f"Generating up to {max_tokens} tokens...")
        print(f"Temperature: {temperature}, Top-k: {top_k}")
        print(f"KV cache: {'enabled' if use_kv_cache else 'disabled'}")
        print()

    # Initialize KV cache if enabled
    kv_cache = None
    if use_kv_cache:
        head_dim = config.n_embd // config.n_head
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=len(prompt_tokens) + max_tokens,
            head_dim=head_dim,
            num_layers=config.n_layer
        )

    # Convert prompt to MLX array
    prompt_ids = mx.array([prompt_tokens], dtype=mx.int32)

    # Prefill phase: process prompt
    start_time = time.time()
    logits = model(prompt_ids, kv_cache=kv_cache)
    mx.eval(logits)
    prefill_time = time.time() - start_time

    if verbose:
        print(f"Prefill time: {prefill_time*1000:.1f}ms")
        print("Generating tokens...")
        print()

    # Generation loop
    generated_tokens = []
    gen_start_time = time.time()

    for i in range(max_tokens):
        # Get logits for last token
        last_logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply top-k filtering if requested
        if top_k is not None:
            k = min(top_k, last_logits.shape[-1])
            top_vals = mx.topk(last_logits, k, axis=-1)
            threshold = mx.min(top_vals, axis=-1, keepdims=True)
            last_logits = mx.where(last_logits >= threshold, last_logits, -float('inf'))

        # Sample next token
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature, axis=-1)
            next_token = mx.random.categorical(probs, num_samples=1)
        else:
            # Greedy sampling
            next_token = mx.argmax(last_logits, axis=-1, keepdims=True)

        # Evaluate to get the token value
        mx.eval(next_token)
        token_id = int(next_token[0, 0])
        generated_tokens.append(token_id)

        if verbose and (i < 10 or (i + 1) % 10 == 0):
            elapsed = time.time() - gen_start_time
            tokens_per_sec = (i + 1) / elapsed
            print(f"Token {i+1:3d}: {token_id:4d} ({tokens_per_sec:.0f} tok/s)")

        # Forward the model with the next token
        next_token_input = next_token  # Already (1, 1)
        logits = model(next_token_input, kv_cache=kv_cache)
        mx.eval(logits)

    gen_time = time.time() - gen_start_time

    if verbose:
        print()
        print(f"Generation complete!")
        print(f"Generated {len(generated_tokens)} tokens in {gen_time:.2f}s")
        print(f"Speed: {len(generated_tokens)/gen_time:.1f} tok/s")

    return generated_tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with nanochat MLX model")
    parser.add_argument("--model-size", type=str, default="d6", choices=['d2', 'd6', 'd10'],
                       help="Model size (d2, d6, d10)")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                       help="Text prompt (will be tokenized as integers)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling (None = sample from full distribution)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable KV cache (slower)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set random seed
    mx.random.seed(args.seed)

    print("=" * 70)
    print("nanochat MLX Inference")
    print("=" * 70)
    print()

    # Create model
    print(f"Loading {args.model_size} model...")
    model, config = create_model(args.model_size)
    print(f"Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print()

    # Tokenize prompt (simple: just use token IDs)
    # In a real implementation, you'd use a proper tokenizer
    # For now, just convert to a list of integers (demo mode)
    prompt_text = args.prompt
    print(f"Prompt: \"{prompt_text}\"")

    # Simple tokenization: use BPE or similar in production
    # For demo, we'll just create random token sequence from vocab
    # In practice, you'd load a tokenizer and use it here
    import hashlib
    hash_obj = hashlib.md5(prompt_text.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    mx.random.seed(hash_int % (2**32))
    prompt_len = min(10, len(prompt_text.split()))
    prompt_tokens = mx.random.randint(0, config.vocab_size, (prompt_len,)).tolist()
    print(f"Note: Using demo tokenization (random tokens)")
    print()

    # Generate text
    generated_tokens = generate_text(
        model,
        config,
        prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=not args.no_cache,
        verbose=True
    )

    print()
    print("=" * 70)
    print("Generated token IDs:")
    print("=" * 70)
    print(f"Prompt: {prompt_tokens}")
    print(f"Generated: {generated_tokens[:20]}{'...' if len(generated_tokens) > 20 else ''}")
    print()
    print("Note: To see actual text, you need to implement a tokenizer")
    print("      and call tokenizer.decode(prompt_tokens + generated_tokens)")


if __name__ == "__main__":
    main()
