"""
MLX Optimizers for nanochat
Wrappers around MLX's built-in optimizers + custom Muon
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math


# Use MLX's built-in AdamW
class AdamW(optim.AdamW):
    """
    AdamW optimizer - wrapper around MLX's built-in
    """
    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)


# Use MLX's built-in SGD
class SGD(optim.SGD):
    """
    SGD with momentum - wrapper around MLX's built-in
    """
    def __init__(self, learning_rate=0.02, momentum=0.95):
        super().__init__(learning_rate=learning_rate, momentum=momentum)


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    This iteration produces something like US'V^T where S' is diagonal with
    S_{ii}' ~ Uniform(0.5, 1.5), which works well in practice.

    Args:
        G: 2D matrix to orthogonalize
        steps: Number of Newton-Schulz iterations (default: 5)

    Returns:
        Orthogonalized matrix
    """
    assert G.ndim == 2, f"G must be 2D, got shape {G.shape}"

    # Coefficients for quintic iteration
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in bfloat16 for stability
    X = G.astype(mx.bfloat16)

    # Handle tall matrices by transposing
    if G.shape[0] > G.shape[1]:
        X = X.T

    # Ensure spectral norm is at most 1
    norm = mx.sqrt(mx.sum(X * X))
    X = X / (norm + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)  # quintic computation
        X = a * X + (B @ X)

    # Transpose back if needed
    if G.shape[0] > G.shape[1]:
        X = X.T

    return X.astype(G.dtype)


class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the
    nearest orthogonal matrix. To efficiently orthogonalize each update, we use a
    Newton-Schulz iteration, which has the advantage that it can be stably run in
    bfloat16 on the GPU.

    Warnings:
    - This optimizer should NOT be used for embedding layers, final FC layer, or any
      0D/1D parameters; those should be optimized by AdamW.
    - Only use for 2D parameters (linear/conv layers).

    Args:
        learning_rate: The learning rate used by the internal SGD
        momentum: The momentum used by the internal SGD
        nesterov: Whether to use Nesterov-style momentum (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use
    """

    def __init__(self, learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    def init_single(self, parameter, state):
        """
        Initialize optimizer state for a single parameter.

        Args:
            parameter: Parameter to initialize state for
            state: State dictionary to populate

        Returns:
            Initialized state dictionary
        """
        state["momentum_buffer"] = mx.zeros_like(parameter)
        return state

    def apply_single(self, gradient, parameter, state):
        """
        Apply Muon update to a single parameter.

        Args:
            gradient: Gradient for this parameter
            parameter: Current parameter value
            state: Optimizer state for this parameter

        Returns:
            Updated parameter
        """
        # Initialize momentum buffer if needed
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = mx.zeros_like(gradient)

        buf = state["momentum_buffer"]

        # Update momentum buffer: buf = momentum * buf + (1 - momentum) * grad
        buf = self.momentum * buf + (1 - self.momentum) * gradient

        # Choose update based on Nesterov flag
        if self.nesterov:
            g = (1 - self.momentum) * gradient + self.momentum * buf
        else:
            g = buf

        # Apply Muon orthogonalization for 2D parameters only
        if parameter.ndim == 2:
            # Orthogonalize the update
            g = zeropower_via_newtonschulz5(g, steps=self.ns_steps)

            # Apply aspect-ratio scaling
            # Scale by sqrt(max(1, rows/cols)) to account for rectangular matrices
            scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
            update = self.learning_rate * scale * g
        else:
            # For non-2D params (shouldn't happen, but handle gracefully)
            update = self.learning_rate * g

        # Update momentum buffer in state
        state["momentum_buffer"] = buf

        # Return updated parameter
        return parameter - update


# Convenience function to create optimizers for GPT model
def create_optimizer(model, learning_rate=0.01, weight_decay=0.0):
    """
    Create AdamW optimizer for GPT model
    Simplified version - uses single optimizer for all parameters

    Args:
        model: GPT model
        learning_rate: Base learning rate
        weight_decay: Weight decay

    Returns:
        optimizer
    """
    # Scale LR by model dimension (similar to original)
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    print(f"Scaling LR by ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

    # Create AdamW optimizer
    optimizer = AdamW(
        learning_rate=learning_rate * dmodel_lr_scale,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=weight_decay
    )

    return optimizer
