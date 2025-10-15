#!/usr/bin/env python3
"""
Inference script with real tokenizer support
Uses tiktoken's GPT-2 tokenizer as a fallback if custom tokenizer not trained
"""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache


def create_model(model_size='d6'):
    """Create GPT model based on size"""
    configs = {
        'd2': GPTConfig(sequence_len=256, vocab_size=50304, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
        'd6': GPTConfig(sequence_len=512, vocab_size=50304, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
        'd10': GPTConfig(sequence_len=1024, vocab_size=50304, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = GPT(config)
    model.init_weights()
    return model, config


def get_tokenizer():
    """
    Get tokenizer. Try custom nanochat tokenizer first, fall back to tiktoken GPT-2.
    """
    try:
        # Try to load custom nanochat tokenizer
        from nanochat.tokenizer import get_tokenizer as get_nanochat_tokenizer
        tokenizer = get_nanochat_tokenizer()
        print("✅ Using nanochat custom tokenizer")
        return tokenizer
    except Exception as e:
        # Fall back to tiktoken GPT-2
        print(f"⚠️  Custom tokenizer not available ({e})")
        print("📦 Using tiktoken gpt2 tokenizer as fallback")
        from nanochat.tokenizer import RustBPETokenizer
        tokenizer = RustBPETokenizer.from_pretrained("gpt2")
        return tokenizer


def generate_text(
    model,
    config,
    tokenizer,
    prompt_text,
    max_tokens=100,
    temperature=1.0,
    top_k=None,
):
    """Generate text from prompt"""
    # Encode prompt
    bos_token_id = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(prompt_text, prepend=bos_token_id)

    print(f"Prompt: \"{prompt_text}\"")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print()

    # Setup KV cache
    head_dim = config.n_embd // config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=len(prompt_tokens) + max_tokens,
        head_dim=head_dim,
        num_layers=config.n_layer
    )

    # Convert to MLX array
    prompt_ids = mx.array([prompt_tokens], dtype=mx.int32)

    # Prefill
    logits = model(prompt_ids, kv_cache=kv_cache)
    mx.eval(logits)

    # Generate
    generated_tokens = []
    for i in range(max_tokens):
        last_logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            k = min(top_k, last_logits.shape[-1])
            top_vals = mx.topk(last_logits, k, axis=-1)
            threshold = mx.min(top_vals, axis=-1, keepdims=True)
            last_logits = mx.where(last_logits >= threshold, last_logits, -float('inf'))

        # Sample
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature, axis=-1)
            next_token = mx.random.categorical(probs, num_samples=1)
        else:
            next_token = mx.argmax(last_logits, axis=-1, keepdims=True)

        mx.eval(next_token)
        token_id = int(next_token[0, 0])
        generated_tokens.append(token_id)

        # Decode and print token
        token_text = tokenizer.decode([token_id])
        print(token_text, end="", flush=True)

        # Forward with next token
        logits = model(next_token, kv_cache=kv_cache)
        mx.eval(logits)

    print("\n")
    return generated_tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with tokenizer support")
    parser.add_argument("--model-size", type=str, default="d6", choices=['d2', 'd6', 'd10'])
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    mx.random.seed(args.seed)

    print("=" * 70)
    print("nanochat MLX Inference (with tokenizer)")
    print("=" * 70)
    print()

    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print()

    # Create model
    print(f"Loading {args.model_size} model...")
    model, config = create_model(args.model_size)
    print(f"Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print()

    # Generate
    generated_tokens = generate_text(
        model,
        config,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("=" * 70)
    print(f"Generated {len(generated_tokens)} tokens")


if __name__ == "__main__":
    main()
