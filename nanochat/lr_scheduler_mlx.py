"""
Learning rate schedulers for MLX training
Includes cosine annealing with warmup, commonly used for transformer training
"""
import math


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler with linear warmup.

    Learning rate schedule:
    1. Warmup: Linear ramp from 0 to max_lr over warmup_steps
    2. Cosine decay: From max_lr to min_lr over remaining steps

    This is the standard schedule used in transformer training (GPT, BERT, etc.)
    """

    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
        """
        Args:
            optimizer: Optimizer instance (AdamW or Muon)
            warmup_steps: Number of warmup steps (linear ramp)
            max_steps: Total training steps (for cosine schedule)
            max_lr: Maximum learning rate (reached after warmup)
            min_lr_ratio: Minimum LR as fraction of max_lr (default: 0.1 = 10%)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.current_step = 0
        self.current_lr = 0.0

    def step(self):
        """
        Update learning rate and increment step counter.
        Should be called once per training iteration.

        Returns:
            current_lr: The new learning rate
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup: lr = max_lr * (step / warmup_steps)
            self.current_lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing after warmup
            # progress = 0.0 at end of warmup, 1.0 at max_steps
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            # Cosine decay: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
            self.current_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                             (1 + math.cos(math.pi * progress))

        # Update optimizer learning rate
        self.optimizer.learning_rate = self.current_lr
        self.current_step += 1

        return self.current_lr

    def get_lr(self):
        """Get current learning rate without stepping."""
        return self.current_lr

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'current_lr': self.current_lr,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.current_lr = state_dict['current_lr']
        self.optimizer.learning_rate = self.current_lr


class ConstantLR:
    """
    Constant learning rate (no scheduling).
    Useful for fine-tuning or debugging.
    """

    def __init__(self, optimizer, lr):
        """
        Args:
            optimizer: Optimizer instance
            lr: Learning rate (constant)
        """
        self.optimizer = optimizer
        self.lr = lr
        self.current_step = 0
        self.optimizer.learning_rate = lr

    def step(self):
        """No-op for constant LR."""
        self.current_step += 1
        return self.lr

    def get_lr(self):
        """Return constant LR."""
        return self.lr

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']


class LinearWarmupOnly:
    """
    Linear warmup followed by constant LR.
    Simpler than cosine annealing, useful for shorter runs.
    """

    def __init__(self, optimizer, warmup_steps, max_lr):
        """
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            max_lr: Maximum learning rate (reached after warmup)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0
        self.current_lr = 0.0

    def step(self):
        """Update LR: linear warmup then constant."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            self.current_lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Constant after warmup
            self.current_lr = self.max_lr

        self.optimizer.learning_rate = self.current_lr
        self.current_step += 1
        return self.current_lr

    def get_lr(self):
        """Get current learning rate."""
        return self.current_lr

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'current_lr': self.current_lr,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.current_lr = state_dict['current_lr']
        self.optimizer.learning_rate = self.current_lr


def visualize_schedule(scheduler_class, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1, num_points=100):
    """
    Visualize a learning rate schedule (for debugging).

    Args:
        scheduler_class: Scheduler class (e.g., CosineAnnealingLR)
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr_ratio: Minimum LR ratio
        num_points: Number of points to sample

    Returns:
        steps: List of step numbers
        lrs: List of learning rates
    """
    # Create dummy optimizer
    class DummyOptimizer:
        def __init__(self):
            self.learning_rate = 0.0

    optimizer = DummyOptimizer()

    # Create scheduler
    if scheduler_class == CosineAnnealingLR:
        scheduler = CosineAnnealingLR(optimizer, warmup_steps, max_steps, max_lr, min_lr_ratio)
    elif scheduler_class == LinearWarmupOnly:
        scheduler = LinearWarmupOnly(optimizer, warmup_steps, max_lr)
    elif scheduler_class == ConstantLR:
        scheduler = ConstantLR(optimizer, max_lr)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_class}")

    # Sample learning rates
    steps = []
    lrs = []
    step_interval = max(1, max_steps // num_points)

    for step in range(0, max_steps, step_interval):
        # Fast forward to step
        while scheduler.current_step < step:
            lr = scheduler.step()

        steps.append(step)
        lrs.append(scheduler.get_lr())

    return steps, lrs


if __name__ == "__main__":
    """
    Test and visualize learning rate schedules.
    """
    print("Learning Rate Schedule Visualization")
    print("=" * 60)

    # Test parameters
    max_steps = 10000
    warmup_steps = 1000
    max_lr = 0.001
    min_lr_ratio = 0.1

    # Test CosineAnnealingLR
    print("\n1. Cosine Annealing with Warmup")
    print(f"   Warmup: {warmup_steps} steps")
    print(f"   Max steps: {max_steps}")
    print(f"   Max LR: {max_lr}")
    print(f"   Min LR: {max_lr * min_lr_ratio}")

    steps, lrs = visualize_schedule(CosineAnnealingLR, warmup_steps, max_steps, max_lr, min_lr_ratio)

    print(f"\n   Sample schedule:")
    for i, (step, lr) in enumerate(zip(steps, lrs)):
        if i % 20 == 0:  # Print every 20th point
            print(f"   Step {step:5d}: LR = {lr:.6f}")

    print(f"\n   Verification:")
    print(f"   - LR at step 0: {lrs[0]:.6f} (should be ~0)")
    print(f"   - LR at step {warmup_steps}: ~{max_lr:.6f}")
    print(f"   - LR at step {max_steps}: ~{max_lr * min_lr_ratio:.6f}")

    # Test LinearWarmupOnly
    print("\n2. Linear Warmup Only")
    steps2, lrs2 = visualize_schedule(LinearWarmupOnly, warmup_steps, max_steps, max_lr)
    print(f"   LR at step {warmup_steps}: {lrs2[warmup_steps//100]:.6f}")
    print(f"   LR at step {max_steps}: {lrs2[-1]:.6f} (constant after warmup)")

    print("\n✅ LR scheduler tests complete!")
