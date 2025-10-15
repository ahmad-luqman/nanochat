# nanochat - Complete Technical Overview

**Author:** Andrej Karpathy
**Repository:** https://github.com/karpathy/nanochat
**Tagline:** The best ChatGPT that $100 can buy

## Table of Contents

1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Model Specifications](#model-specifications)
6. [Installation Guide](#installation-guide)
7. [Running the Pipeline](#running-the-pipeline)
8. [Using the Trained Model](#using-the-trained-model)
9. [Advanced Configuration](#advanced-configuration)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Hardware Requirements](#hardware-requirements)

---

## Overview

nanochat is a complete, minimal implementation of a ChatGPT-like Large Language Model (LLM) trained from scratch. It demonstrates the entire ML pipeline in a single, clean, hackable codebase designed to run on a single 8×H100 node for approximately $100.

### What It Includes

The project covers the complete LLM lifecycle:

1. **Tokenization** - Custom BPE tokenizer implementation in Rust
2. **Pretraining** - Base language model training on 11.2B tokens
3. **Midtraining** - Teaching conversation formatting and special tokens
4. **Supervised Fine-tuning (SFT)** - Domain adaptation for chat
5. **Reinforcement Learning** - Optional RL on GSM8K (math reasoning)
6. **Inference** - CLI and web-based chat interfaces
7. **Evaluation** - Comprehensive benchmarking on multiple tasks

### Key Characteristics

- **Minimal**: ~8,300 lines of code across 44 files
- **Complete**: End-to-end pipeline from raw data to ChatGPT-like UI
- **Educational**: Designed as the capstone project for LLM101n course
- **Accessible**: Trains in ~4 hours for ~$100 on cloud GPUs
- **Hackable**: Clean, readable code without excessive abstractions

---

## Technical Architecture

### High-Level Pipeline

```
Raw Text Data (FineWeb)
    ↓
[Tokenizer Training] → BPE Tokenizer (vocab: 65,536)
    ↓
[Base Training] → Pretrained Model (561M params, 11.2B tokens)
    ↓
[Midtraining] → Conversation-Aware Model
    ↓
[Supervised Fine-tuning] → Chat-Optimized Model
    ↓
[Optional: RL] → RL-Enhanced Model (GSM8K)
    ↓
[Inference] → Web UI / CLI Chat Interface
```

### Core Components

**1. Tokenizer (`rustbpe/`)**
- High-performance Byte Pair Encoding (BPE) in Rust
- Python bindings via PyO3/Maturin
- Vocabulary size: 65,536 (2^16)
- Trained on ~2B characters
- Compression ratio: ~4.8 chars/token

**2. Model Architecture (`nanochat/gpt.py`)**
- Transformer-based decoder architecture
- Configurable depth (layers)
- Default: d20 = 20 layers, 561M parameters
- Supports scaling to d26 (GPT-2 grade) and beyond

**3. Training Engine (`nanochat/engine.py`)**
- Multi-GPU distributed training via PyTorch DDP
- Mixed precision training
- Gradient accumulation support
- Checkpoint management
- WandB integration for experiment tracking

**4. Optimizers**
- **AdamW** (`nanochat/adamw.py`) - Standard optimizer
- **Muon** (`nanochat/muon.py`) - Custom optimizer variant

**5. Data Pipeline**
- **Dataset** (`nanochat/dataset.py`) - Downloads FineWeb shards
- **DataLoader** (`nanochat/dataloader.py`) - Efficient batching
- Distributed data loading across GPUs

**6. Evaluation Framework**
- **CORE** - Pretraining benchmark
- **ARC-Challenge/Easy** - Reasoning tasks
- **GSM8K** - Math word problems
- **HumanEval** - Code generation
- **MMLU** - Multitask knowledge
- **ChatCORE** - Conversation quality

---

## Technology Stack

### Languages

- **Python 3.10+** - Main implementation language
- **Rust** - High-performance tokenizer (Edition 2024)
- **HTML/CSS/JavaScript** - Web UI template

### Core Dependencies

```toml
[project.dependencies]
datasets >= 4.0.0          # HuggingFace datasets
fastapi >= 0.117.1         # Web API framework
fastapi >= 0.117.1         # Web API framework
numpy == 1.26.4            # Numerical computing
psutil >= 7.1.0            # System utilities
regex >= 2025.9.1          # Regular expressions
tiktoken >= 0.11.0         # OpenAI tokenizer (reference)
tokenizers >= 0.22.0       # HuggingFace tokenizers
torch >= 2.8.0             # Deep learning framework (CUDA 12.8)
uvicorn >= 0.36.0          # ASGI server
wandb >= 0.21.3            # Experiment tracking
```

### Rust Dependencies (rustbpe)

```toml
[dependencies]
dary_heap = "0.3"          # Priority queue
indexmap = "2.2"           # Ordered hashmaps
fancy-regex = "0.16.1"     # Advanced regex
log = "0.4.28"             # Logging
pyo3 = "0.23.3"            # Python bindings
pyo3-log = "0.12.4"        # Python logging integration
ahash = "0.8.12"           # Fast hashing
rayon = "1.11.0"           # Parallel iterators
compact_str = "0.9.0"      # Optimized strings
```

### Build System

- **uv** - Fast Python package installer and environment manager
- **maturin** - Rust-Python build tool
- **cargo** - Rust package manager

### PyTorch Configuration

- CUDA version: 12.8
- Custom index: `https://download.pytorch.org/whl/cu128`
- Multi-GPU via `torch.distributed` (DDP)
- Uses `torchrun` for launching distributed training

---

## Project Structure

```
nanochat/
├── nanochat/                    # Core library module
│   ├── __init__.py
│   ├── gpt.py                  # GPT model architecture
│   ├── engine.py               # Training engine & loops
│   ├── tokenizer.py            # Tokenizer interface
│   ├── dataset.py              # Data downloading utilities
│   ├── dataloader.py           # Batching & data loading
│   ├── adamw.py                # AdamW optimizer
│   ├── muon.py                 # Muon optimizer variant
│   ├── execution.py            # Code execution for evals
│   ├── core_eval.py            # CORE benchmark evaluation
│   ├── loss_eval.py            # Loss evaluation utilities
│   ├── checkpoint_manager.py   # Model checkpoint handling
│   ├── configurator.py         # Configuration management
│   ├── common.py               # Shared utilities
│   ├── report.py               # Report generation
│   ├── ui.html                 # Web UI template
│   └── logo.svg                # Logo asset
│
├── rustbpe/                     # Rust BPE tokenizer
│   ├── Cargo.toml              # Rust package manifest
│   ├── Cargo.lock              # Dependency lock file
│   ├── README.md               # Tokenizer documentation
│   └── src/
│       └── lib.rs              # Tokenizer implementation
│
├── scripts/                     # Training & inference scripts
│   ├── tok_train.py            # Train tokenizer
│   ├── tok_eval.py             # Evaluate tokenizer
│   ├── base_train.py           # Pretrain base model
│   ├── base_eval.py            # Evaluate base on CORE
│   ├── base_loss.py            # Base model loss evaluation
│   ├── mid_train.py            # Midtraining (conversation)
│   ├── chat_sft.py             # Supervised fine-tuning
│   ├── chat_rl.py              # Reinforcement learning
│   ├── chat_eval.py            # Chat model evaluation
│   ├── chat_cli.py             # CLI chat interface
│   └── chat_web.py             # Web chat interface
│
├── tasks/                       # Evaluation task implementations
│   ├── common.py               # Shared task utilities
│   ├── arc.py                  # ARC benchmark
│   ├── gsm8k.py                # GSM8K math reasoning
│   ├── humaneval.py            # HumanEval code generation
│   ├── mmlu.py                 # MMLU knowledge
│   └── smoltalk.py             # SmolTalk conversation
│
├── tests/                       # Test suite
│   └── test_rustbpe.py         # Tokenizer tests
│
├── dev/                         # Development utilities
│   └── nanochat.png            # Logo image
│
├── speedrun.sh                  # Full pipeline script (~$100, 4hrs)
├── pyproject.toml               # Python project configuration
├── uv.lock                      # Python dependency lock
├── .python-version              # Python version specification
├── .gitignore                   # Git ignore rules
└── README.md                    # Main documentation
```

### Data Storage

**Default location:** `~/.cache/nanochat/` (configurable via `$NANOCHAT_BASE_DIR`)

```
~/.cache/nanochat/
├── data/                        # Training data shards
│   ├── shard_000000.txt.gz     # ~100MB each
│   ├── shard_000001.txt.gz
│   └── ...                      # Up to 240 shards for d20
├── tokenizer/                   # Trained tokenizer files
├── models/                      # Model checkpoints
│   ├── base/
│   ├── mid/
│   ├── sft/
│   └── rl/
├── eval_bundle/                 # Evaluation datasets (~162MB)
└── report/                      # Generated report sections
```

---

## Model Specifications

### Default Model (d20)

**Architecture:**
- **Parameters:** 561M
- **Layers (depth):** 20
- **Vocabulary:** 65,536 tokens
- **Context window:** Configurable (typically 2048-4096)
- **Architecture type:** Decoder-only Transformer

**Training:**
- **Total tokens:** 11.2B (Chinchilla optimal: 20× parameters)
- **Data source:** FineWeb (curated web text)
- **Data volume:** ~240 shards × 250M chars = ~24GB
- **Training time:** ~4 hours on 8×H100
- **Cost:** ~$96 ($24/hr × 4 hours)
- **Batch size:** Configurable per device (default: 32)

**Performance (Example Results):**
```
| Metric          | BASE   | MID    | SFT    | RL     |
|-----------------|--------|--------|--------|--------|
| CORE            | 0.2219 | -      | -      | -      |
| ARC-Challenge   | -      | 0.2875 | 0.2807 | -      |
| ARC-Easy        | -      | 0.3561 | 0.3876 | -      |
| GSM8K           | -      | 0.0250 | 0.0455 | 0.0758 |
| HumanEval       | -      | 0.0671 | 0.0854 | -      |
| MMLU            | -      | 0.3111 | 0.3151 | -      |
| ChatCORE        | -      | 0.0730 | 0.0884 | -      |
```

### Larger Model Configurations

**d26 Model (~$300, 12 hours)**
- **Parameters:** ~800M+ (exact count varies)
- **Training tokens:** ~16B
- **Data shards needed:** 450
- **Batch size adjustment:** 16 (vs 32 for d20)
- **Performance:** Slightly outperforms GPT-2 on CORE

**d32+ Model (~$1000, 41.6 hours)**
- Experimental, not fully documented
- Requires further batch size and data adjustments

---

## Installation Guide

### Prerequisites

**Hardware:**
- **Recommended:** 8×H100 GPUs (80GB VRAM each)
- **Alternative:** 8×A100 GPUs (80GB VRAM each, slightly slower)
- **Minimum:** 1×GPU with 80GB VRAM (8× slower, same results)
- **Lower VRAM:** Possible with `--device_batch_size` tuning

**Software:**
- Linux (Ubuntu 20.04+) or macOS
- Git
- Curl
- Bash shell
- Internet connection (for downloading dependencies and data)

**Cloud Providers:**
- Lambda Labs (recommended by author)
- AWS, GCP, Azure (P4/P5 instances)
- RunPod, Vast.ai, etc.

### Step-by-Step Installation

**1. Clone the Repository**

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

**2. Automatic Setup via speedrun.sh**

The script handles all dependencies automatically:

```bash
bash speedrun.sh
```

**What gets installed:**

1. **uv** (Python package manager)
   - Installed via: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. **Python Virtual Environment**
   - Created in `.venv/`
   - Python dependencies from `pyproject.toml`

3. **Rust & Cargo**
   - Installed via: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

4. **rustbpe Tokenizer**
   - Compiled with: `uv run maturin develop --release`

5. **Training Data**
   - Downloads ~24GB of FineWeb shards
   - Stored in `~/.cache/nanochat/data/`

6. **Evaluation Bundle**
   - Downloads ~162MB eval_bundle.zip
   - Extracts to `~/.cache/nanochat/eval_bundle/`

### Manual Installation (Optional)

If you prefer manual setup:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install Python dependencies
uv sync

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download data
python -m nanochat.dataset -n 240

# Download eval bundle
curl -L -o eval_bundle.zip \
  https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip eval_bundle.zip
mv eval_bundle ~/.cache/nanochat/
```

---

## Running the Pipeline

### Quick Start: Full Pipeline

**Simple launch:**
```bash
bash speedrun.sh
```

**With screen session (recommended):**
```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

**Screen commands:**
- Detach: `Ctrl-a d`
- Reattach: `screen -r speedrun`
- View log: `tail -f speedrun.log`
- Kill: `screen -X -S speedrun quit`

**With WandB logging:**
```bash
# First login
wandb login

# Then run
WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### Pipeline Stages (in speedrun.sh)

**Stage 1: Setup (5-10 minutes)**
```bash
# Environment variables
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# uv installation & venv setup
uv venv && source .venv/bin/activate && uv sync

# Rust installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

**Stage 2: Tokenizer (30-45 minutes)**
```bash
# Build rustbpe
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download initial data (8 shards for tokenizer)
python -m nanochat.dataset -n 8

# Download full data in background (240 shards total)
python -m nanochat.dataset -n 240 &

# Train tokenizer on 2B characters
python -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer
python -m scripts.tok_eval
```

**Stage 3: Base Training/Pretraining (~2 hours)**
```bash
# Download evaluation bundle
curl -L -o eval_bundle.zip \
  https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip && mv eval_bundle $NANOCHAT_BASE_DIR

# Wait for data download
wait $DATASET_DOWNLOAD_PID

# Train base model (d20, 561M params, 11.2B tokens)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
  -- --depth=20 --run=$WANDB_RUN

# Evaluate loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

# Evaluate CORE benchmark
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**Stage 4: Midtraining (~30 minutes)**
```bash
# Teach conversation formatting, special tokens, tool use
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
  -- --run=$WANDB_RUN

# Evaluate mid model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

**Stage 5: Supervised Fine-tuning (~30 minutes)**
```bash
# Domain adaptation for chat
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
  -- --run=$WANDB_RUN

# Evaluate SFT model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Stage 6: Reinforcement Learning (Optional, ~30 minutes)**
```bash
# RL on GSM8K (commented out by default)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
  -- --run=$WANDB_RUN

# Evaluate RL model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval \
  -- -i rl -a GSM8K
```

**Stage 7: Report Generation**
```bash
# Generate markdown report with all metrics
python -m nanochat.report generate

# Report saved to: ./report.md
```

### Running Individual Stages

Activate the environment first:
```bash
source .venv/bin/activate
```

**Train only tokenizer:**
```bash
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

**Train only base model:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

**Train only midtraining:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
```

**Train only SFT:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

**Evaluate specific benchmark:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K
```

---

## Using the Trained Model

### Prerequisites

Ensure virtual environment is active:
```bash
source .venv/bin/activate
```

### Web UI (Recommended)

**Start the server:**
```bash
python -m scripts.chat_web
```

**Access the UI:**
- Local: `http://localhost:8000`
- Remote/Cloud: `http://<PUBLIC_IP>:8000`
  - On Lambda: Use public IP from dashboard
  - Example: `http://209.20.xxx.xxx:8000`

**Features:**
- ChatGPT-like interface
- Real-time streaming responses
- Conversation history
- Clean, minimal design

### CLI Interface

**Interactive mode:**
```bash
python -m scripts.chat_cli
```

**Single prompt:**
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"
```

**Example prompts:**
```bash
# Creative writing
python -m scripts.chat_cli -p "Write a short poem about machine learning"

# Reasoning
python -m scripts.chat_cli -p "Explain why the sky appears blue"

# Math (if RL trained)
python -m scripts.chat_cli -p "What is 15% of 240?"

# Code generation
python -m scripts.chat_cli -p "Write a Python function to reverse a string"
```

### View Training Report

```bash
cat report.md
```

**Report includes:**
- System information
- Training timestamps
- Tokenizer metrics (compression ratio)
- Model architecture details
- Evaluation results on all benchmarks
- Total wall clock time
- Codebase statistics

---

## Advanced Configuration

### Training Different Model Sizes

**d26 Model (GPT-2 grade, ~$300, 12 hours):**

Edit `speedrun.sh`:

```bash
# Line 66: Download more data (450 shards instead of 240)
python -m nanochat.dataset -n 450 &

# Line 95: Train d26 with reduced batch size (prevent OOM)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
  -- --depth=26 --device_batch_size=16

# Line 105: Match batch size in midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
  -- --device_batch_size=16
```

**Calculating data requirements:**
```python
# For any depth d:
# 1. Get parameter count (varies by depth)
# 2. Tokens needed = params × 20 (Chinchilla optimal)
# 3. Characters needed = tokens × 4.8 (avg compression)
# 4. Shards needed = characters / 250M
```

### Memory Management

**If you run out of VRAM, reduce batch size:**

```bash
# Try these values in order: 32 → 16 → 8 → 4 → 2 → 1
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
  -- --depth=20 --device_batch_size=16
```

**What happens:**
- Training speed decreases (more gradient accumulation steps)
- Results remain identical (same effective batch size)
- More sequential compute, less parallel compute

### Single GPU Training

**Remove torchrun, run directly:**

```bash
# Instead of:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Use:
python -m scripts.base_train -- --depth=20
```

**Trade-offs:**
- 8× slower training time
- Same final results (gradient accumulation compensates)
- ~32 hours instead of 4 hours for d20

### Custom Data Directory

```bash
export NANOCHAT_BASE_DIR="/path/to/custom/dir"
bash speedrun.sh
```

### WandB Configuration

**Login once:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

**Run with custom name:**
```bash
WANDB_RUN=my_experiment_name bash speedrun.sh
```

**Disable WandB:**
```bash
# Default behavior (WANDB_RUN not set)
bash speedrun.sh
# Uses "dummy" run name, skips wandb logging
```

### Other Hardware Platforms

**A100 GPUs (Ampere):**
- Works out of the box
- Slightly slower than H100
- Same configuration

**Lower VRAM GPUs (< 80GB):**
- Reduce `--device_batch_size`
- May need to reduce model size

**Apple Silicon (MPS):**
- Not supported out of the box
- Requires code modifications (device placement)

**Intel XPU / AMD ROCm:**
- Not supported out of the box
- Vanilla PyTorch code should be portable with effort

---

## Evaluation Metrics

### Benchmarks

**CORE (Pretraining)**
- Evaluates base language modeling capability
- Used during pretraining
- Score: ~0.22 for d20

**ARC (Abstract Reasoning Corpus)**
- **ARC-Challenge:** Harder questions
- **ARC-Easy:** Easier questions
- Tests reasoning and world knowledge
- Scores: ~0.28-0.29 (Challenge), ~0.36-0.39 (Easy)

**GSM8K (Math Word Problems)**
- Grade school math problems
- Tests arithmetic and reasoning
- Improves significantly with RL
- Scores: 0.025 (mid) → 0.046 (sft) → 0.076 (rl)

**HumanEval (Code Generation)**
- Python function completion
- Tests programming ability
- Score: ~0.067-0.085

**MMLU (Multitask Language Understanding)**
- Broad knowledge across domains
- Multiple choice questions
- Score: ~0.31

**ChatCORE (Conversation Quality)**
- Evaluates chat-specific capabilities
- Applied after midtraining
- Score: ~0.073-0.088

### Running Evaluations

**Evaluate specific checkpoint:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Evaluate on specific benchmark:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K
```

**Multiple benchmarks:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC GSM8K
```

---

## Hardware Requirements

### GPU Memory Requirements

**8×H100 (80GB each) - Recommended**
- Total VRAM: 640GB
- Batch size: 32 per device
- Training time: ~4 hours (d20)
- Cost: ~$96 @ $24/hr

**8×A100 (80GB each) - Supported**
- Total VRAM: 640GB
- Batch size: 32 per device
- Training time: ~5 hours (d20)
- Slightly slower than H100

**1×H100/A100 (80GB)**
- Total VRAM: 80GB
- Batch size: 32 (with gradient accumulation)
- Training time: ~32 hours (d20)
- Cost: ~$96 @ $3/hr

**Lower VRAM GPUs**
- 40GB VRAM: Reduce batch size to 8-16
- 24GB VRAM: Reduce batch size to 2-4, possibly reduce model size
- 16GB VRAM: Train smaller models only (d10-d15)

### CPU & System Requirements

**CPU:**
- Multi-core recommended (16+ cores ideal)
- Used for data loading, preprocessing

**RAM:**
- Minimum: 64GB
- Recommended: 128GB+
- Needed for data buffering

**Storage:**
- Minimum: 100GB free
- Recommended: 200GB+
- Breakdown:
  - Data shards: ~24GB (d20) to ~45GB (d26)
  - Model checkpoints: ~2-5GB per checkpoint
  - Eval bundle: ~162MB
  - Virtual env & dependencies: ~5GB

**Network:**
- Fast internet for initial downloads (~25GB)
- No special requirements during training

### Cloud Provider Recommendations

**Lambda Labs** (Author's choice)
- 8×H100 node: ~$24/hr
- Simple setup, reliable
- Good documentation

**AWS**
- p5.48xlarge: 8×H100, expensive
- p4d.24xlarge: 8×A100

**GCP**
- a3-highgpu-8g: 8×H100
- a2-highgpu-8g: 8×A100

**Azure**
- ND H100 v5 series

**Budget Options**
- RunPod, Vast.ai: Spot instances
- Variable pricing, less reliable

---

## Testing

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/test_rustbpe.py -v -s
```

### Test Configuration

From `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `--device_batch_size`
- Try: 32 → 16 → 8 → 4 → 2 → 1

**2. CUDA/GPU not detected**
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Check CUDA drivers installed
- Check PyTorch CUDA version matches system CUDA

**3. Data download fails**
- Check internet connection
- Retry: `python -m nanochat.dataset -n 240`
- Downloads resume automatically

**4. Rust compilation fails**
- Ensure Rust installed: `rustc --version`
- Reinstall: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

**5. Permission errors**
- Check `$NANOCHAT_BASE_DIR` permissions
- Default: `~/.cache/nanochat` should be user-writable

**6. Port already in use (web UI)**
- Change port: Edit `scripts/chat_web.py`
- Or kill existing process: `lsof -ti:8000 | xargs kill`

---

## Advanced Topics

### Asking Questions About the Code

**Using files-to-prompt:**
```bash
# Install
pip install files-to-prompt

# Package entire repo
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh \
  --ignore "*target*" --cxml > packaged.txt

# Now paste packaged.txt to your favorite LLM
```

**Using DeepWiki:**
- Visit: https://deepwiki.com/karpathy/nanochat
- (Change github.com → deepwiki.com in URL)

### Code Statistics

From typical run:
- **Total characters:** ~334,000
- **Total lines:** ~8,300
- **Total files:** 44
- **Approximate tokens:** ~83,500
- **Dependencies:** ~2,000 lines (uv.lock)

### Philosophy

nanochat is NOT:
- A configurable framework
- Exhaustively parameterized
- Production-grade serving system
- State-of-the-art performance maximizer

nanochat IS:
- A single, cohesive implementation
- Minimal, readable, hackable
- A strong educational baseline
- Maximally forkable for research
- Complete end-to-end demonstration

---

## Contributing

The goal is to improve micro-model state-of-the-art at <$1000 budgets while maintaining:

1. **Accessibility** - Low cost and cognitive complexity
2. **Minimalism** - No giant config objects or factories
3. **Readability** - Clean, hackable code
4. **Completeness** - End-to-end pipeline

---

## Acknowledgements

- **nanoGPT** - Previous project covering pretraining only
- **modded-nanoGPT** - Gamification inspiration, metrics
- **HuggingFace** - FineWeb and SmolTalk datasets
- **Lambda Labs** - Compute for development
- **Alec Radford** - LLM guidance and advice

---

## Citation

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

---

## License

MIT

---

## Quick Reference

### Essential Commands

```bash
# Full pipeline
bash speedrun.sh

# Web chat
source .venv/bin/activate && python -m scripts.chat_web

# CLI chat
source .venv/bin/activate && python -m scripts.chat_cli

# View report
cat report.md

# Run tests
pytest tests/test_rustbpe.py -v
```

### Important Paths

- Code: `/path/to/nanochat/`
- Data: `~/.cache/nanochat/` (or `$NANOCHAT_BASE_DIR`)
- Logs: `speedrun.log` (if using screen)
- Report: `./report.md`

### Key Files

- **speedrun.sh** - Full pipeline script
- **pyproject.toml** - Python dependencies
- **nanochat/gpt.py** - Model architecture
- **nanochat/engine.py** - Training loops
- **scripts/chat_web.py** - Web interface

---

**Document generated:** October 15, 2025
**Repository:** https://github.com/karpathy/nanochat
**Author:** Andrej Karpathy
