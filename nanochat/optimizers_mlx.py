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


# Use MLX's built-in SGD for now (will add custom Muon later if needed)
class SGD(optim.SGD):
    """
    SGD with momentum - wrapper around MLX's built-in
    """
    def __init__(self, learning_rate=0.02, momentum=0.95):
        super().__init__(learning_rate=learning_rate, momentum=momentum)


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
