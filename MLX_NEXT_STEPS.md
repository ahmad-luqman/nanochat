# nanochat MLX Port - Next Steps

**Current Status**: Core functionality complete (training, inference, optimizers, KV cache)

**Goal**: Achieve full feature parity with original PyTorch implementation

---

## Plan A: Minimal but Functional (Priority 1)

**Goal**: Train and chat with a real model on actual data (FineWebEdu)

**Timeline**: ~5 hours

### Tasks

#### 1. Checkpoint System (1 hour)
**Purpose**: Save/load trained models for inference and continuation

**Files to create:**
- `nanochat/checkpoint_mlx.py` - Checkpoint manager for MLX
- `test_checkpoint_mlx.py` - Test save/load

**Features:**
- Save model weights to disk (use `mx.save`)
- Load model weights (use `mx.load`)
- Save optimizer state
- Load optimizer state for training continuation
- Checkpoint versioning and metadata

**API:**
```python
# Save checkpoint
save_checkpoint(model, optimizer, step, loss, path)

# Load checkpoint
model, optimizer, meta = load_checkpoint(path)
```

**Testing:**
- Save d2 model, load, verify weights identical
- Save mid-training, load, continue training
- Test checkpoint with both AdamW and Muon

---

#### 2. Tokenizer Training Setup (1 hour)
**Purpose**: Train custom BPE tokenizer on data

**Files to modify:**
- Build rustbpe Rust module (already exists)
- Verify `scripts/tok_train.py` works without PyTorch

**Steps:**
1. Build rustbpe: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`
2. Download sample data: `python -m nanochat.dataset -n 8`
3. Train tokenizer: `python -m scripts.tok_train --max_chars=2000000000`
4. Verify tokenizer works: `python -m scripts.tok_eval`

**Output:**
- `~/.cache/nanochat/tokenizer/tokenizer.pkl` - Trained tokenizer
- Vocab size: 2^16 = 65,536 tokens
- Compression ratio: ~4.5-5.0 chars/token

---

#### 3. Real Data Pipeline (2 hours)
**Purpose**: Load and preprocess FineWebEdu data for training

**Files to create:**
- `nanochat/dataset_mlx.py` - MLX data loader for FineWebEdu
- `test_dataset_mlx.py` - Test data loading

**Features:**
- Download FineWebEdu shards (already implemented in `nanochat/dataset.py`)
- Read tokenized shards efficiently
- Batch loading with proper padding/truncation
- Support for distributed training (future)

**API:**
```python
# Create data loader
dataloader = FineWebDataLoader(
    data_dir="~/.cache/nanochat/data",
    tokenizer=tokenizer,
    batch_size=8,
    seq_len=1024,
    num_shards=240
)

# Iterate
for batch in dataloader:
    x, y = batch  # input tokens, target tokens
    loss = model(x, targets=y)
```

**Changes needed:**
- Port `nanochat/dataset.py` data loading (remove PyTorch tensor conversion)
- Return MLX arrays instead of PyTorch tensors
- Handle sharding and shuffling

---

#### 4. Simple Chat CLI (1 hour)
**Purpose**: Interactive chat interface with trained model

**Files to create:**
- `scripts/chat_cli_mlx.py` - Chat CLI for MLX models

**Features:**
- Load trained checkpoint
- Interactive REPL for chatting
- Streaming generation (token-by-token)
- Conversation history
- Special token handling (<|user_start|>, <|assistant_start|>, etc.)
- Optional: Calculator tool integration

**API:**
```bash
# Chat interactively
python scripts/chat_cli_mlx.py --checkpoint out/d6_final.npz

# Single prompt
python scripts/chat_cli_mlx.py \
    --checkpoint out/d6_final.npz \
    --prompt "Why is the sky blue?"
```

**Implementation:**
- Use `tokenizer.render_conversation()` for proper formatting
- Use KV cache for fast generation
- Pretty print with color coding
- Ctrl+C to exit gracefully

---

### Testing Plan A

**Test 1: Checkpoint System**
```bash
# Train for 100 steps, save checkpoint
python scripts/train_mlx.py --model-size d2 --max-steps 100
# Load checkpoint, continue for 50 more steps
python scripts/train_mlx.py --model-size d2 --max-steps 50 --resume out/checkpoint_100.npz
```

**Test 2: Tokenizer Training**
```bash
# Build rustbpe
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
# Download data (8 shards = ~2B chars)
python -m nanochat.dataset -n 8
# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000
# Verify
python -m scripts.tok_eval
```

**Test 3: Real Data Training**
```bash
# Train d6 on FineWebEdu for 1000 steps
python scripts/train_mlx.py \
    --model-size d6 \
    --batch-size 8 \
    --seq-len 512 \
    --max-steps 1000 \
    --real-data \
    --checkpoint-dir out/d6_run1
```

**Test 4: Chat with Trained Model**
```bash
# Interactive chat
python scripts/chat_cli_mlx.py --checkpoint out/d6_run1/final.npz

# Single prompt
python scripts/chat_cli_mlx.py \
    --checkpoint out/d6_run1/final.npz \
    --prompt "Explain quantum computing in simple terms"
```

---

### Success Criteria for Plan A

- ✅ Can save and load model checkpoints without data loss
- ✅ Custom tokenizer trained on FineWebEdu (vocab size 65k)
- ✅ Can train d6 model on real FineWebEdu data
- ✅ Training loss decreases over time (not random)
- ✅ Can chat interactively with trained model
- ✅ Model generates coherent text (not gibberish)

**Deliverables:**
- Checkpoint system working
- Tokenizer trained and saved
- Real data loader implemented
- d6 model trained for 1k-5k steps on FineWebEdu
- Chat CLI working with trained model

**Result**: You have a working chatbot trained on real data! 🎉

---

## Plan B: Full Feature Parity (Priority 2)

**Goal**: Port all remaining nanochat features for complete compatibility

**Timeline**: ~15-20 hours

### Phase 1: Evaluation Suite (3 hours)

#### 1.1 Loss Evaluation
**File**: `scripts/base_loss_mlx.py`

**Features:**
- Evaluate perplexity on train/val splits
- Calculate bits-per-byte (BPB)
- Generate sample text for qualitative analysis
- Compare before/after training

**Testing:**
```bash
python scripts/base_loss_mlx.py --checkpoint out/d6_final.npz
```

#### 1.2 CORE Benchmark
**File**: `scripts/base_eval_mlx.py`

**Features:**
- Port CORE benchmark evaluation
- Download eval_bundle.zip (~162MB)
- Run model on multiple-choice tasks
- Calculate accuracy metrics
- Compare to baseline models

**Tasks in CORE:**
- HellaSwag (commonsense reasoning)
- PIQA (physical reasoning)
- WinoGrande (coreference resolution)
- ARC-Easy/Challenge (science questions)
- And more...

**Testing:**
```bash
# Download eval bundle
curl -L -o eval_bundle.zip \
    https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip eval_bundle.zip -d ~/.cache/nanochat/

# Run evaluation
python scripts/base_eval_mlx.py --checkpoint out/d6_final.npz
```

#### 1.3 Chat Evaluation
**File**: `scripts/chat_eval_mlx.py`

**Features:**
- Evaluate chat models on conversation tasks
- MT-Bench style evaluation
- GSM8K math reasoning
- Calculate win rates

**Testing:**
```bash
python scripts/chat_eval_mlx.py --checkpoint out/chat_sft_final.npz
```

---

### Phase 2: Advanced Training (6 hours)

#### 2.1 Midtraining
**File**: `scripts/mid_train_mlx.py`

**Purpose**: Teach model conversation format and tool use

**Features:**
- Train on conversation data with special tokens
- Teach <|user_start|>, <|assistant_start|>, etc.
- Teach calculator tool usage (<|python_start|>, <|output_start|>)
- Short training run (~5k steps)

**Data:**
- Mix of base pretraining data
- Conversation-formatted data
- Tool use examples

**Testing:**
```bash
python scripts/mid_train_mlx.py \
    --checkpoint out/d6_base_final.npz \
    --output out/d6_mid.npz \
    --steps 5000
```

#### 2.2 Supervised Finetuning (SFT)
**File**: `scripts/chat_sft_mlx.py`

**Purpose**: Adapt model to specific conversation style

**Features:**
- Train on high-quality conversation data
- Mask user turns (only train on assistant responses)
- Use `render_conversation()` for proper formatting
- Domain adaptation per conversation

**Data:**
- Curated conversation datasets
- Each example is a full conversation
- Mask tokens where mask=0

**Testing:**
```bash
python scripts/chat_sft_mlx.py \
    --checkpoint out/d6_mid.npz \
    --output out/d6_sft.npz \
    --steps 10000
```

#### 2.3 Reinforcement Learning (Optional)
**File**: `scripts/chat_rl_mlx.py`

**Purpose**: Further improve via RL (GRPO)

**Features:**
- Group Relative Policy Optimization (GRPO)
- Train on GSM8K math problems
- Reward = correctness of solution
- Multiple samples per prompt

**Complexity**: High - advanced RL algorithm

**Testing:**
```bash
python scripts/chat_rl_mlx.py \
    --checkpoint out/d6_sft.npz \
    --output out/d6_rl.npz \
    --task gsm8k
```

---

### Phase 3: Interactive Tools (3 hours)

#### 3.1 Web UI
**File**: `scripts/chat_web_mlx.py`

**Purpose**: ChatGPT-style web interface

**Features:**
- Beautiful web UI (likely uses Flask/FastAPI)
- Streaming generation
- Conversation history
- Model switching
- Temperature/top-k controls
- Copy/share conversations

**Tech stack:**
- Backend: FastAPI + MLX
- Frontend: HTML/CSS/JS (probably existing in repo)
- WebSockets for streaming

**Testing:**
```bash
python scripts/chat_web_mlx.py --checkpoint out/d6_sft.npz --port 8000
# Open browser: http://localhost:8000
```

#### 3.2 Enhanced CLI
**File**: Enhance `scripts/chat_cli_mlx.py`

**Features:**
- Colored output (user vs assistant)
- Calculator tool integration
- Conversation export/import
- Multi-turn context
- /help, /clear, /reset commands
- Configurable generation params

---

### Phase 4: Model Management (2 hours)

#### 4.1 Model Hub Integration
**Features:**
- Download pre-trained models from HuggingFace
- Upload trained models
- Model cards and metadata
- Version management

#### 4.2 Conversion Utilities
**Features:**
- Convert PyTorch checkpoints → MLX
- Convert MLX checkpoints → PyTorch
- Support for different model formats
- Quantization (8-bit, 4-bit)

#### 4.3 Deployment Tools
**Features:**
- Model serving API
- Batched inference
- Caching and optimization
- Monitoring and logging

---

### Phase 5: Documentation & Polish (2 hours)

#### 5.1 Documentation
- Complete API documentation
- Training guide (how to train from scratch)
- Evaluation guide
- Deployment guide
- Troubleshooting section

#### 5.2 Examples
- Example notebooks
- Training scripts for different scenarios
- Finetuning examples
- Integration examples

#### 5.3 Testing
- Comprehensive test suite
- Integration tests
- Performance benchmarks
- Regression tests

---

## Testing Plan B

### Full Pipeline Test
```bash
# 1. Train base model
python scripts/base_train_mlx.py --depth 6 --steps 50000

# 2. Evaluate base model
python scripts/base_loss_mlx.py
python scripts/base_eval_mlx.py

# 3. Midtraining
python scripts/mid_train_mlx.py --steps 5000

# 4. Evaluate after midtraining
python scripts/chat_eval_mlx.py -i mid

# 5. Supervised finetuning
python scripts/chat_sft_mlx.py --steps 10000

# 6. Evaluate after SFT
python scripts/chat_eval_mlx.py -i sft

# 7. (Optional) RL training
python scripts/chat_rl_mlx.py --task gsm8k

# 8. Chat with final model
python scripts/chat_cli_mlx.py
python scripts/chat_web_mlx.py
```

### Success Criteria for Plan B

**Training:**
- ✅ Base pretraining works on full FineWebEdu dataset
- ✅ Midtraining teaches conversation format
- ✅ SFT improves chat quality
- ✅ (Optional) RL improves math reasoning

**Evaluation:**
- ✅ Loss evaluation shows improvement over training
- ✅ CORE benchmark scores comparable to PyTorch version
- ✅ Chat evaluation shows good conversation quality

**Interactive:**
- ✅ Web UI works and looks good
- ✅ CLI supports all features
- ✅ Calculator tool works correctly

**Compatibility:**
- ✅ All original speedrun.sh features ported
- ✅ Same model quality as PyTorch version
- ✅ Same or better performance on M4 Max

---

## Resource Requirements

### Plan A
- **Time**: ~5 hours
- **Disk**: ~25GB (data + checkpoints)
- **RAM**: 64GB recommended for d6, 128GB for d10
- **Compute**: ~2-4 hours training time for d6 (5k steps)

### Plan B
- **Time**: ~15-20 hours
- **Disk**: ~50GB (full data + multiple checkpoints)
- **RAM**: 128GB recommended for d20 training
- **Compute**: ~24 hours for full d20 training (50k steps)

---

## Priority Order

1. **Plan A** (Do first) - Get end-to-end working
2. **Plan B Phase 1** - Evaluation suite
3. **Plan B Phase 3** - Interactive tools (web UI)
4. **Plan B Phase 2** - Advanced training
5. **Plan B Phase 4-5** - Polish and deployment

---

## Notes

### Framework Differences to Handle

Most code is framework-agnostic! These components need minimal changes:

**Already Compatible:**
- ✅ Tokenizer (tiktoken/rustbpe) - pure Python
- ✅ Dataset downloading - file I/O
- ✅ Special token handling - string manipulation
- ✅ Evaluation logic - mostly NumPy

**Need MLX Porting:**
- Data loading (PyTorch → MLX arrays)
- Loss computation (already done)
- Model forward pass (already done)
- Optimizer updates (already done)

**Estimated Porting Effort:**
- Most files: 10-20% needs changes
- Some files: 100% compatible (just imports)

---

## Current Branch Status

**Branch**: `mlx-port`

**Commits**: 10 commits
- Planning, documentation
- Phase 1: Model architecture
- Phase 2: Optimizers (AdamW, Muon)
- Phase 2: KV cache
- Phase 3-4: Data loader, training script
- Phase 5: Inference script
- Phase 6: Speedrun script
- Tokenizer integration

**Ready for**: Plan A implementation

---

## Timeline Estimate

### Plan A: Minimal but Functional
- Week 1: Complete Plan A (5 hours)
- Result: Working chatbot on real data

### Plan B: Full Feature Parity
- Week 2-3: Complete Plan B (15-20 hours)
- Result: Feature-complete nanochat on MLX

**Total**: 20-25 hours for complete port
