"""
Data loading utilities for MLX training
Includes both synthetic and real data loaders
"""
import mlx.core as mx
import numpy as np
import os
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common_mlx import get_base_dir


class SyntheticDataLoader:
    """
    Synthetic data loader that generates random token sequences
    Useful for testing training loops without real data
    """

    def __init__(self, vocab_size, batch_size, seq_len, num_batches=100):
        """
        Args:
            vocab_size: Size of vocabulary
            batch_size: Batch size
            seq_len: Sequence length
            num_batches: Number of batches to generate per epoch
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Generate random token sequences
        # Input: tokens [0:T-1], Target: tokens [1:T]
        tokens = mx.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len + 1))
        x = tokens[:, :-1]  # input tokens
        y = tokens[:, 1:]   # target tokens (shifted by 1)

        self.current_batch += 1
        return x, y

    def __len__(self):
        return self.num_batches


class TextDataLoader:
    """
    Real data loader that streams from parquet files and tokenizes on-the-fly.
    Yields batches of (input_tokens, target_tokens) as MLX arrays.
    """

    def __init__(self, split="train", batch_size=32, seq_len=1024, start=0, step=1):
        """
        Args:
            split: "train" or "val"
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            start: Starting row_group index (for DDP)
            step: Step size for row_groups (for DDP)
        """
        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.start = start
        self.step = step

        # Load tokenizer
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Token buffer for creating sequences
        self.token_buffer = []

    def __iter__(self):
        """
        Iterate through the dataset, yielding batches of token sequences
        """
        self.token_buffer = []

        # Get text batches from parquet files
        for text_batch in parquets_iter_batched(self.split, self.start, self.step):
            # Tokenize all texts in this batch
            for text in text_batch:
                tokens = self.tokenizer.encode(text)
                self.token_buffer.extend(tokens)

                # Yield batches whenever we have enough tokens
                while len(self.token_buffer) >= self.batch_size * (self.seq_len + 1):
                    yield self._create_batch()

        # Yield remaining tokens if we have enough for at least one sequence
        while len(self.token_buffer) >= self.seq_len + 1:
            yield self._create_batch()

    def _create_batch(self):
        """
        Create a batch of (input, target) sequences from the token buffer.
        Each sequence is seq_len+1 tokens long, then split into input and target.
        """
        batch_tokens = []

        for _ in range(self.batch_size):
            if len(self.token_buffer) < self.seq_len + 1:
                # Not enough tokens for a full batch, pad with zeros
                seq = self.token_buffer + [0] * (self.seq_len + 1 - len(self.token_buffer))
                self.token_buffer = []
            else:
                # Extract seq_len+1 tokens
                seq = self.token_buffer[:self.seq_len + 1]
                self.token_buffer = self.token_buffer[self.seq_len + 1:]

            batch_tokens.append(seq)

        # Convert to MLX array
        tokens = mx.array(batch_tokens, dtype=mx.int32)

        # Split into input (0:T-1) and target (1:T)
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        return x, y
