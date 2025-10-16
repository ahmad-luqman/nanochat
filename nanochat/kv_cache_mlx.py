"""
KV Cache for efficient inference with MLX
Ported from PyTorch implementation in engine.py
"""
import mlx.core as mx


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.

    Usage:
        # Create cache
        cache = KVCache(batch_size=1, num_heads=6, seq_len=1024, head_dim=64, num_layers=6)

        # In model forward pass
        for layer_idx, block in enumerate(self.h):
            # ... compute k, v ...
            k, v = cache.insert_kv(layer_idx, k, v)  # returns full cache view
            # ... use k, v for attention ...
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        Initialize KV cache structure.

        Args:
            batch_size: Batch size
            num_heads: Number of KV heads (not query heads)
            seq_len: Initial sequence length capacity
            head_dim: Head dimension
            num_layers: Number of transformer layers
        """
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        # We stack them as (num_layers, 2, B, H, T, D) where index 1 is [0=K, 1=V]
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in time in the cache

    def reset(self):
        """Reset cache position to 0"""
        self.pos = 0

    def get_pos(self):
        """Get current cache position"""
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.

        Args:
            other: Another KVCache to copy from
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, 2 (k/v), num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"

        # 2) initialize the cache with same dtype
        self.kv_cache = mx.zeros(self.kv_shape, dtype=other.kv_cache.dtype)

        # 3) copy the data over (broadcast if batch size differs)
        if self.kv_shape[2] != other.kv_shape[2]:
            # Need to broadcast batch dimension
            other_data = other.kv_cache[:, :, :, :, :other.pos, :]
            # Broadcast from (L, 2, 1, H, T, D) to (L, 2, B, H, T, D)
            other_data = mx.broadcast_to(other_data,
                                        (self.kv_shape[0], self.kv_shape[1], self.kv_shape[2],
                                         other.kv_shape[3], other.pos, other.kv_shape[5]))
            self.kv_cache[:, :, :, :, :other.pos, :] = other_data
        else:
            self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache[:, :, :, :, :other.pos, :]

        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v, debug=False):
        """
        Insert new key/value tensors into the cache and return full cache view.

        Args:
            layer_idx: Layer index (0-indexed)
            k: Key tensor of shape (B, H, T_add, D)
            v: Value tensor of shape (B, H, T_add, D)
            debug: If True, print debug information

        Returns:
            key_view: Full cached keys up to current position (B, H, T_total, D)
            value_view: Full cached values up to current position (B, H, T_total, D)
        """
        # Lazy initialize the cache here because we need to know the dtype
        if self.kv_cache is None:
            self.kv_cache = mx.zeros(self.kv_shape, dtype=k.dtype)

        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.shape
        t0, t1 = self.pos, self.pos + T_add

        if debug and layer_idx == 0:
            print(f"  [Layer {layer_idx}] pos={self.pos}, t0={t0}, t1={t1}, T_add={T_add}")
            print(f"  [Layer {layer_idx}] k.shape={k.shape}, cache.shape={self.kv_cache.shape}")

        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.shape[4]:
            # MLX doesn't support in-place resize, so we need to create a new array
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023  # round up to nearest multiple of 1024

            # Create new larger cache
            new_shape = list(self.kv_cache.shape)
            new_shape[4] = t_needed
            new_cache = mx.zeros(new_shape, dtype=self.kv_cache.dtype)

            # Copy old data
            old_t = self.kv_cache.shape[4]
            new_cache[:, :, :, :, :old_t, :] = self.kv_cache

            # Replace cache
            self.kv_cache = new_cache

        # Store k, v in the cache using slicing
        # Note: In MLX, we need to rebuild the array to update it
        # Get current layer's cache
        layer_cache = self.kv_cache[layer_idx]  # (2, B, H, T_cache, D)
        layer_k = layer_cache[0]  # (B, H, T_cache, D)
        layer_v = layer_cache[1]  # (B, H, T_cache, D)

        # Update with new k, v at position t0:t1
        # Build updated arrays by concatenating parts
        if debug and layer_idx == 0:
            print(f"  [Layer {layer_idx}] Concatenating: [:, :, :{t0}, :] + k + [:, :, {t1}:, :]")
            print(f"  [Layer {layer_idx}] layer_k[:,:,:t0,:].shape = {layer_k[:, :, :t0, :].shape}")
            print(f"  [Layer {layer_idx}] k.shape = {k.shape}")
            print(f"  [Layer {layer_idx}] layer_k[:,:,t1:,:].shape = {layer_k[:, :, t1:, :].shape}")

        updated_k = mx.concatenate([
            layer_k[:, :, :t0, :],
            k,
            layer_k[:, :, t1:, :]
        ], axis=2)
        updated_v = mx.concatenate([
            layer_v[:, :, :t0, :],
            v,
            layer_v[:, :, t1:, :]
        ], axis=2)

        if debug and layer_idx == 0:
            print(f"  [Layer {layer_idx}] updated_k.shape = {updated_k.shape}")

        # Rebuild the cache with updated layer
        # Create list of layers
        new_layers = []
        for l_idx in range(self.kv_cache.shape[0]):
            if l_idx == layer_idx:
                # Use updated k, v for this layer
                new_layers.append(mx.stack([updated_k, updated_v], axis=0))
            else:
                # Keep existing k, v for other layers
                new_layers.append(self.kv_cache[l_idx])

        # Stack all layers back together
        self.kv_cache = mx.stack(new_layers, axis=0)

        # Return the full cached keys/values up to current position (as a slice)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]

        if debug and layer_idx == 0:
            print(f"  [Layer {layer_idx}] Returning key_view.shape = {key_view.shape}")

        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.shape[0] - 1:
            self.pos = t1

        return key_view, value_view
