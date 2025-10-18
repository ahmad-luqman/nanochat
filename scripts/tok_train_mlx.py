"""
Train a tokenizer using the RustBPE library (MLX-compatible version).
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import mlx.core as mx
import numpy as np

from nanochat.tokenizer import RustBPETokenizer
from nanochat.common_mlx import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
print("\nTraining tokenizer...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text
print("✅ Tokenizer sanity check passed")

# -----------------------------------------------------------------------------
# Cache token bytes for bits-per-byte evaluation
# This mapping allows us to report loss in bits per byte, which is invariant
# to the vocab size of the tokenizer.
print("\nCaching token bytes...")
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]  # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)

# Save as MLX array
token_bytes = mx.array(token_bytes, dtype=mx.int32)
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes_mlx.npz")
mx.savez(token_bytes_path, token_bytes=token_bytes)
print(f"Saved token_bytes to {token_bytes_path}")

# Compute statistics (use numpy for boolean indexing, MLX doesn't support it yet)
token_bytes_np = np.array(token_bytes)
token_bytes_nonzero_np = token_bytes_np[token_bytes_np > 0]
token_bytes_nonzero = mx.array(token_bytes_nonzero_np, dtype=mx.float32)
print(f"\nTokenizer statistics:")
print(f"  Vocab size: {vocab_size:,}")
print(f"  Special tokens: {len(special_set)}")
print(f"  Token bytes (min/max/mean/std): "
      f"{int(mx.min(token_bytes_nonzero).item())}/{int(mx.max(token_bytes_nonzero).item())}/"
      f"{mx.mean(token_bytes_nonzero).item():.2f}/{mx.std(token_bytes_nonzero).item():.2f}")

print(f"\n✅ Tokenizer training complete!")
print(f"   Saved to: {tokenizer_dir}")
print(f"   Training time: {train_time:.2f}s")
