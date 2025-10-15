"""
Test script for MLX TextDataLoader
"""
import time
from nanochat.data_mlx import TextDataLoader

print("Testing TextDataLoader with real data...")
print("-" * 60)

# Create data loader (small batch for testing)
loader = TextDataLoader(
    split="train",
    batch_size=4,
    seq_len=256,
)

print(f"Configuration:")
print(f"  Split: train")
print(f"  Batch size: 4")
print(f"  Sequence length: 256")
print(f"  Vocab size: {loader.vocab_size:,}")
print()

# Test loading a few batches
print("Loading first 3 batches...")
t0 = time.time()

for i, (x, y) in enumerate(loader):
    if i >= 3:
        break

    print(f"\nBatch {i+1}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Input range: {x.min().item()} - {x.max().item()}")
    print(f"  Target range: {y.min().item()} - {y.max().item()}")

    # Decode first sequence to verify it's real text
    first_seq = x[0].tolist()
    decoded = loader.tokenizer.decode(first_seq[:50])  # First 50 tokens
    print(f"  Sample text: {decoded[:100]}...")

t1 = time.time()
print(f"\n✅ Data loading test passed!")
print(f"   Time for 3 batches: {t1-t0:.2f}s")
