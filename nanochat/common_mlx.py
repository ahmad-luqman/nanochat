"""
Common utilities for nanochat MLX port.
Simplified version without PyTorch/CUDA dependencies.
"""

import os
import re
import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    """Get the base directory for nanochat data"""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def print0(s="", **kwargs):
    """Print only from rank 0 (for MLX, we're always rank 0)"""
    print(s, **kwargs)


def print_banner():
    """Print nanochat banner"""
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)


def is_ddp():
    """MLX version - no distributed training for now"""
    return False


def get_dist_info():
    """MLX version - single device, no distributed"""
    return False, 0, 0, 1  # not ddp, rank 0, local_rank 0, world_size 1


def compute_init():
    """Basic initialization for MLX"""
    import mlx.core as mx

    # Reproducibility
    mx.random.seed(42)

    # No distributed setup needed for MLX single device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if ddp_rank == 0:
        logger.info(f"MLX device: {mx.default_device()}")
        logger.info(f"World size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, mx.default_device()


def compute_cleanup():
    """Companion function to compute_init"""
    # No cleanup needed for MLX
    pass


class DummyWandb:
    """Dummy wandb for when we don't want to use it"""
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass
