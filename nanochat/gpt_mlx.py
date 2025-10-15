"""
GPT model (MLX port)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference

Ported to MLX for Apple Silicon by Claude
"""

import math
from functools import partial
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Import from MLX common utilities
from nanochat.common_mlx import get_dist_info, print0


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    # MLX rms_norm requires weight parameter, so pass None for no learnable params
    # Actually, let's implement it manually since MLX's signature is different
    eps = 1e-5
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings"""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = mx.concatenate([y1, y2], axis=3)  # re-assemble
    out = out.astype(x.dtype)  # ensure input/output dtypes match
    return out


def repeat_kv(x, n_rep):
    """Repeat key/value heads for multi-query attention"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    # MLX doesn't have repeat_interleave, so we do it manually
    x = mx.expand_dims(x, axis=2)  # (bs, n_kv_heads, 1, slen, head_dim)
    x = mx.broadcast_to(x, (bs, n_kv_heads, n_rep, slen, head_dim))
    x = x.reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def __call__(self, x, cos_sin, kv_cache):
        B, T, C = x.shape

        # Project the input to get queries, keys, and values
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        # Transpose to make head be batch dim: (B, T, H, D) -> (B, H, T, D)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.shape[2]  # number of queries in this forward pass
        Tk = k.shape[2]  # number of keys/values in total

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # Attention: queries attend to keys/values autoregressively
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # MLX has mx.fast.scaled_dot_product_attention
            mask = mx.triu(mx.ones((Tq, Tk), dtype=mx.bool_), k=1)  # upper triangular = True
            y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0 / math.sqrt(self.head_dim))
        elif Tq == 1:
            # During inference but with a single query
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim))
        else:
            # During inference with a chunk of queries
            mask = mx.zeros((Tq, Tk), dtype=mx.bool_)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                mask[:, :prefix_len] = True  # attend to prefix
            # Causal attention within chunk
            chunk_mask = mx.tril(mx.ones((Tq, Tq), dtype=mx.bool_))
            mask[:, prefix_len:] = chunk_mask
            y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0 / math.sqrt(self.head_dim))

        # Re-assemble the heads side by side and project back
        y = mx.transpose(y, (0, 2, 1, 3))  # (B, H, T, D) -> (B, T, H, D)
        y = y.reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.relu(x) ** 2  # relu^2 activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create embedding and blocks
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # In MLX, we can just store them as instance attributes (no need for register_buffer)
        self.cos = cos
        self.sin = sin

        # Cast embeddings to bfloat16
        self.wte.weight = self.wte.weight.astype(mx.bfloat16)

    def init_weights(self):
        """Initialize weights using custom scheme"""
        self._init_weights_recursive(self)
        # Zero out classifier weights
        self.lm_head.weight = mx.zeros_like(self.lm_head.weight)
        # Zero out c_proj weights in all blocks
        for block in self.h:
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
        # Re-init rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights_recursive(self, module):
        """Recursively initialize weights"""
        if isinstance(module, nn.Linear):
            # Custom initialization scheme
            fan_out, fan_in = module.weight.shape
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            module.weight = mx.random.normal(module.weight.shape, scale=std)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = mx.zeros_like(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(module.weight.shape, scale=1.0)
        # Recurse for child modules
        elif hasattr(module, 'children'):
            for child in module.children():
                self._init_weights_recursive(child)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Precompute rotary embeddings"""
        # Stride the channels
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Stride the time steps
        t = mx.arange(seq_len, dtype=mx.float32)
        # Calculate rotation frequencies at each (time, channel) pair
        freqs = mx.outer(t, inv_freq)  # (seq_len, head_dim//2)
        cos, sin = mx.cos(freqs), mx.sin(freqs)
        cos, sin = cos.astype(mx.bfloat16), sin.astype(mx.bfloat16)
        # Add batch and head dims for broadcasting: (1, seq_len, 1, head_dim//2)
        cos = mx.expand_dims(mx.expand_dims(cos, 0), 2)
        sin = mx.expand_dims(mx.expand_dims(sin, 0), 2)
        return cos, sin

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model"""
        # Count parameters recursively
        def count_params(tree):
            total = 0
            if isinstance(tree, dict):
                for v in tree.values():
                    total += count_params(v)
            elif isinstance(tree, list):
                for v in tree:
                    total += count_params(v)
            elif hasattr(tree, 'size'):
                total += tree.size
            return total

        nparams = count_params(self.parameters())
        nparams_embedding = self.wte.weight.size
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """Setup optimizers (will be ported later)"""
        # Placeholder for now - will implement MLX optimizers in Phase 2
        raise NotImplementedError("Optimizers will be ported in Phase 2")

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.shape

        # Grab the rotary embeddings for current sequence length
        assert T <= self.cos.shape[1], f"Sequence length {T} exceeds rotary cache {self.cos.shape[1]}"

        # If kv cache exists, offset the rotary embeddings
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # Training mode: compute and return the loss
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)  # logits softcap
            logits = logits.astype(mx.float32)  # use float32 for logits
            # MLX cross entropy
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            # Mask out ignore_index (-1)
            # MLX doesn't support boolean indexing, so we use where to zero out invalid positions
            mask = (targets_flat != -1).astype(mx.float32)
            valid_count = mx.sum(mask)
            # Compute loss for all positions
            ce_loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
            # Zero out invalid positions and compute mean over valid positions
            ce_loss = ce_loss * mask
            if loss_reduction == 'mean':
                loss = mx.sum(ce_loss) / mx.maximum(valid_count, 1)  # avoid div by zero
            else:
                loss = ce_loss
            return loss
        else:
            # Inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)
            return logits

    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        Assumes batch size is 1 and tokens is a list of ints.
        """
        assert isinstance(tokens, list)
        mx.random.seed(seed)

        ids = mx.array([tokens], dtype=mx.int32)  # add batch dim
        for _ in range(max_tokens):
            logits = self(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if top_k is not None:
                # Top-k filtering
                # MLX topk only returns values, so we use argpartition
                k = min(top_k, logits.shape[-1])
                top_vals = mx.topk(logits, k)
                # Set logits below the k-th largest value to -inf
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                logits = mx.where(logits >= threshold, logits, -float('inf'))

            if temperature > 0:
                logits = logits / temperature
                probs = mx.softmax(logits, axis=-1)
                next_ids = mx.random.categorical(probs, num_samples=1)
            else:
                next_ids = mx.argmax(logits, axis=-1, keepdims=True)

            ids = mx.concatenate([ids, next_ids], axis=1)
            token = int(next_ids[0, 0])
            yield token
