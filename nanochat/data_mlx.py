"""
Data loading utilities for MLX training
Simple synthetic data loader for testing and debugging
"""
import mlx.core as mx
import numpy as np


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
    Text data loader for real training
    TODO: Implement in Phase 3 for real data loading
    """

    def __init__(self, data_path, vocab_size, batch_size, seq_len):
        """
        Args:
            data_path: Path to tokenized data file
            vocab_size: Size of vocabulary
            batch_size: Batch size
            seq_len: Sequence length
        """
        raise NotImplementedError("Real data loading will be implemented in Phase 3")
