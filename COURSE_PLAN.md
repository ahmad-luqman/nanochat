# LLM Training Course Plan - NanoChat Edition

> **A hands-on learning course to understand how Large Language Models work**
> Using the nanochat codebase as your playground

---

## 🎯 Course Overview

**Goal**: Understand how LLMs work from scratch by building, training, and using one.

**Approach**: Learn by doing - read code, modify it, run experiments, see results.

**Prerequisites**: Basic Python, some understanding of neural networks helpful but not required.

**Estimated Time**: 20-30 hours over 2-3 weeks

---

## 📚 Course Modules

### Module 1: The Big Picture - What is an LLM? (ELI5)

#### 🧒 ELI5 Explanation

**Imagine you have a very smart parrot:**
- The parrot has read millions of books and websites
- When you say something, it predicts what word should come next
- It does this over and over to create sentences
- The parrot learned patterns by practicing on all those books

**That's basically an LLM!**
- Instead of a parrot, it's a math model (neural network)
- Instead of reading, it's "trained" on text data
- Instead of remembering everything, it learns patterns
- It predicts one word (token) at a time to generate text

#### 📖 Concepts Covered

1. **Tokens**: Words broken into pieces (like LEGO blocks of language)
2. **Training**: Teaching the model to predict next tokens
3. **Generation**: Using the trained model to create new text
4. **Stages**: Pretraining → Fine-tuning → Deployment

#### 🔍 Code References

- **Architecture Overview**: `nanochat/gpt_mlx.py:1-50` (GPT class and config)
- **Simple Example**: Look at how text flows through the system

#### ✏️ Exercises

1. **Exercise 1.1**: Run a pretrained model and observe output
   ```bash
   python scripts/chat_cli_mlx.py --checkpoint checkpoints/d10_sft/best.npz --prompt "Hello, how are you?" --max-tokens 50
   ```
   - Try different prompts
   - Observe how it predicts one token at a time
   - Notice how temperature changes the randomness

2. **Exercise 1.2**: Compare base model vs SFT model
   ```bash
   # Base model (pretrained only)
   python scripts/chat_cli_mlx.py --checkpoint checkpoints/d10_pretrain_causal_fix/best.npz --mode plain --prompt "The capital of France is"

   # SFT model (instruction-tuned)
   python scripts/chat_cli_mlx.py --checkpoint checkpoints/d10_sft/best.npz --mode chat --prompt "What is the capital of France?"
   ```
   - Notice the difference in response style
   - Understand why SFT makes models better at following instructions

#### 📊 Visualization Exercise

Draw a diagram showing:
```
User Input → Tokenization → Model → Token Prediction → Detokenization → Output Text
```

---

### Module 2: Tokenization - From Text to Numbers

#### 🧒 ELI5 Explanation

**Imagine you're sending a secret message using a codebook:**
- You have a dictionary with 65,536 "codes" (tokens)
- Each code can be a full word, part of a word, or a special symbol
- To send "Hello world", you look up codes: [15496, 1879]
- The receiver uses the same codebook to decode it back

**That's tokenization!**
- Computers can't understand text, only numbers
- Tokenizer converts text → numbers (encoding)
- Numbers → text (decoding)
- Special tokens mark things like "start of conversation" or "user is speaking"

#### 📖 Concepts Covered

1. **BPE (Byte Pair Encoding)**: How the tokenizer was trained
2. **Vocabulary**: The 65,536 possible tokens
3. **Special Tokens**: `<|bos|>`, `<|user_start|>`, `<|assistant_end|>`, etc.
4. **Token IDs**: Numbers that represent tokens (0-65535)
5. **Conversation Formatting**: How chat messages are structured

#### 🔍 Code References

- **Tokenizer Interface**: `nanochat/tokenizer.py:1-100`
- **Encoding/Decoding**: `nanochat/tokenizer.py:150-200`
- **Conversation Rendering**: `nanochat/tokenizer.py:250-350`
- **Special Tokens**: `nanochat/tokenizer.py:100-150`

#### ✏️ Exercises

**Exercise 2.1**: Explore the tokenizer
```python
# Create this file: exercises/ex_2_1_tokenizer_basics.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os

# Load tokenizer
tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# Experiment 1: Encode simple text
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Experiment 2: Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Experiment 3: See individual tokens
for token_id in tokens:
    token_text = tokenizer.decode([token_id])
    print(f"  Token {token_id}: '{token_text}'")

# TODO: Try different texts and see how they're tokenized
# - Long words vs short words
# - Numbers
# - Special characters
# - Different languages
```

**Exercise 2.2**: Understanding special tokens
```python
# Create this file: exercises/ex_2_2_special_tokens.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os

tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# Special tokens
print("Special Tokens:")
print(f"  BOS (Beginning of Sequence): {tokenizer.get_bos_token_id()}")
print(f"  User Start: {tokenizer.encode_special('<|user_start|>')}")
print(f"  User End: {tokenizer.encode_special('<|user_end|>')}")
print(f"  Assistant Start: {tokenizer.encode_special('<|assistant_start|>')}")
print(f"  Assistant End: {tokenizer.encode_special('<|assistant_end|>')}")

# Format a conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}

tokens, mask = tokenizer.render_conversation(conversation)
print(f"\nConversation tokens: {tokens}")
print(f"Loss mask: {mask}")
print(f"\nDecoded:\n{tokenizer.decode(tokens)}")

# TODO: Create your own conversation and see how it's tokenized
```

**Exercise 2.3**: Token efficiency
```python
# Create this file: exercises/ex_2_3_token_efficiency.py
# Compare how different texts are tokenized

texts = [
    "Hello",
    "Supercalifragilisticexpialidocious",
    "123456789",
    "人工智能",  # Chinese
    "🚀🌟💡",  # Emojis
    "The quick brown fox jumps over the lazy dog",
]

for text in texts:
    tokens = tokenizer.encode(text)
    chars = len(text)
    token_count = len(tokens)
    ratio = chars / token_count
    print(f"Text: '{text}'")
    print(f"  Characters: {chars}, Tokens: {token_count}, Ratio: {ratio:.2f}")
    print(f"  Tokens: {tokens}\n")

# TODO: Why do some texts tokenize more efficiently than others?
```

#### 🎯 Learning Outcomes

- Understand how text becomes numbers
- Know what tokens are and why they matter
- Recognize special tokens in the model's output
- Understand conversation formatting for chat models

---

### Module 3: The Neural Network Architecture

#### 🧒 ELI5 Explanation

**Imagine a chain of smart boxes (Transformer layers):**

1. **Embedding Box**: Converts token numbers into "meaning vectors"
   - Each token gets a 512-dimensional vector (like GPS coordinates in 512D space)
   - Similar words get similar coordinates

2. **Transformer Boxes** (10 of them stacked):
   - Each box has two main parts:
     - **Attention**: "Look at previous words to understand context"
     - **Feed-Forward**: "Think about what this means"
   - Information flows through all 10 boxes, getting smarter each time

3. **Output Box**: Converts final vectors back to token predictions
   - Outputs 65,536 numbers (one per possible token)
   - Highest number = most likely next token

#### 📖 Concepts Covered

1. **Embeddings**: Converting tokens to vectors
2. **Transformer Layers**: The core processing units
3. **Multi-Head Attention**: Looking at context in different ways
4. **Feed-Forward Networks**: Processing individual positions
5. **Layer Normalization**: Keeping numbers stable
6. **Residual Connections**: Allowing gradient flow
7. **Causal Masking**: Ensuring we only look at previous tokens
8. **KV Cache**: Optimization for fast generation

#### 🔍 Code References

**Core Architecture**:
- **GPTConfig**: `nanochat/gpt_mlx.py:15-30` - Model hyperparameters
- **GPT Model**: `nanochat/gpt_mlx.py:160-220` - Main model class
- **Embeddings**: `nanochat/gpt_mlx.py:180-185` - Token + position embeddings
- **Transformer Block**: `nanochat/gpt_mlx.py:70-130` - Single layer
- **Attention**: `nanochat/gpt_mlx.py:90-110` - Multi-head attention
- **Output Head**: `nanochat/gpt_mlx.py:210-215` - Final predictions

**Supporting Code**:
- **KV Cache**: `nanochat/kv_cache_mlx.py:1-100` - Fast generation optimization
- **Weight Initialization**: `nanochat/gpt_mlx.py:145-160` - How weights start

#### ✏️ Exercises

**Exercise 3.1**: Explore model architecture
```python
# Create: exercises/ex_3_1_model_architecture.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import mlx.core as mx

# Create a tiny model
config = GPTConfig(
    sequence_len=256,
    vocab_size=65536,
    n_layer=2,        # Only 2 layers
    n_head=4,         # 4 attention heads
    n_kv_head=4,
    n_embd=128        # 128-dimensional embeddings
)

model = GPT(config)
model.init_weights()

print("Model Configuration:")
print(f"  Layers: {config.n_layer}")
print(f"  Attention heads: {config.n_head}")
print(f"  Embedding dimension: {config.n_embd}")
print(f"  Vocabulary size: {config.vocab_size}")
print(f"  Max sequence length: {config.sequence_len}")

# Count parameters
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

nparams = count_params(model.parameters())
print(f"\nTotal parameters: {nparams:,} ({nparams/1e6:.2f}M)")

# TODO: Change n_layer, n_head, n_embd and see how parameter count changes
# TODO: Calculate: Why does doubling n_embd more than double parameters?
```

**Exercise 3.2**: Watch data flow through the model
```python
# Create: exercises/ex_3_2_forward_pass.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import mlx.core as mx

# Small model
config = GPTConfig(sequence_len=256, vocab_size=65536, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)
model = GPT(config)
model.init_weights()

# Create dummy input: batch of 1, sequence of 10 tokens
input_tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32)
print(f"Input shape: {input_tokens.shape}")  # (1, 10)

# Forward pass
logits = model(input_tokens)
print(f"Output logits shape: {logits.shape}")  # (1, 10, 65536)

# Interpret the output
print(f"\nFor each of 10 positions, we have 65536 scores (one per token)")
print(f"Last position predictions (top 5 token IDs):")
last_position_logits = logits[0, -1, :]  # Shape: (65536,)
top_5_indices = mx.argpartition(-last_position_logits, 5)[:5]
print(f"  Top token IDs: {top_5_indices.tolist()}")

# TODO: Try different sequence lengths
# TODO: Try batch_size > 1
# TODO: Look at logits for different positions
```

**Exercise 3.3**: Understanding attention
```python
# Create: exercises/ex_3_3_attention.py
# Visualize what attention is doing (conceptually)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This is a conceptual exercise - write your understanding:

print("Attention Mechanism:")
print("\n1. QUERY: 'What should I focus on?'")
print("2. KEY: 'What information do I have?'")
print("3. VALUE: 'What is that information?'")
print("\n4. Attention scores = similarity(Query, Keys)")
print("5. Output = weighted sum of Values\n")

# Example: Predicting next word in "The cat sat on the"
print("Example: Predicting next word in 'The cat sat on the ___'")
print("\nWhen processing 'the' (last position):")
print("  - Attends strongly to: 'sat', 'on' (recent context)")
print("  - Attends weakly to: 'The', 'cat' (distant context)")
print("  - Predicts: 'mat', 'floor', 'chair' (likely completions)")

# TODO: Think about why we need MULTIPLE attention heads (n_head=8)
# TODO: Each head can focus on different aspects!
#       - Head 1: grammar patterns
#       - Head 2: semantic meaning
#       - Head 3: positional relationships
#       - etc.
```

**Exercise 3.4**: Experimenting with KV cache
```python
# Create: exercises/ex_3_4_kv_cache.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache
import mlx.core as mx
import time

config = GPTConfig(sequence_len=512, vocab_size=65536, n_layer=4, n_head=6, n_kv_head=6, n_embd=384)
model = GPT(config)
model.init_weights()

# Generate WITHOUT KV cache (slow - recomputes everything)
print("Without KV cache (recomputing all positions each step):")
prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
start = time.time()

for i in range(20):
    logits = model(prompt)
    next_token = mx.argmax(logits[0, -1, :])
    prompt = mx.concatenate([prompt, next_token[None, None]], axis=1)

no_cache_time = time.time() - start
print(f"  Generated 20 tokens in {no_cache_time:.3f}s")

# Generate WITH KV cache (fast - only computes new position)
print("\nWith KV cache (only computing new positions):")
kv_cache = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=config.sequence_len,
    head_dim=config.n_embd // config.n_head,
    num_layers=config.n_layer
)

prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
start = time.time()

# First pass: process prompt
logits = model(prompt, kv_cache=kv_cache)
mx.eval(logits)

for i in range(20):
    next_token = mx.argmax(logits[0, -1, :])
    next_token_input = next_token[None, None]
    logits = model(next_token_input, kv_cache=kv_cache)
    mx.eval(logits)

cache_time = time.time() - start
print(f"  Generated 20 tokens in {cache_time:.3f}s")
print(f"\nSpeedup: {no_cache_time/cache_time:.1f}x faster!")

# TODO: Try generating 100 tokens and see the difference
# TODO: Understand why cache makes such a big difference
```

#### 🎯 Learning Outcomes

- Understand the transformer architecture
- Know what each component does
- Appreciate why LLMs need so many parameters
- Understand the importance of KV cache for efficiency

---

### Module 4: Training - Teaching the Model

#### 🧒 ELI5 Explanation

**Imagine teaching someone to finish sentences:**

**Training process:**
1. Show them: "The cat sat on the ___"
2. They guess: "tree" (wrong!)
3. You say: "No, it was 'mat'"
4. They adjust their brain a tiny bit to do better next time
5. Repeat 10,000,000 times with different sentences

**That's model training!**
- Data: Millions of text examples
- Loss: How wrong the prediction was
- Backpropagation: Adjusting weights to reduce loss
- Optimizer: The algorithm that does the adjusting
- Epochs: Going through all the data

#### 📖 Concepts Covered

1. **Loss Function**: Measuring how wrong predictions are
2. **Cross-Entropy Loss**: The specific loss for language modeling
3. **Backpropagation**: Computing gradients
4. **Optimizers**: AdamW, Muon (different ways to update weights)
5. **Learning Rate**: How big each adjustment is
6. **Learning Rate Scheduling**: Changing LR over time
7. **Gradient Accumulation**: Training with large effective batch sizes
8. **Checkpointing**: Saving model weights
9. **Validation**: Measuring performance on unseen data

#### 🔍 Code References

**Training Scripts**:
- **Pretraining**: `scripts/train_mlx.py:1-400` - Main training loop
- **SFT Training**: `scripts/sft_mlx.py:200-400` - Fine-tuning loop

**Core Components**:
- **Loss Calculation**: `scripts/train_mlx.py:180-200`
- **Masked Loss (SFT)**: `scripts/sft_mlx.py:173-200`
- **Optimizer Setup**: `nanochat/optimizers_mlx.py:1-100`
- **LR Scheduler**: `nanochat/lr_scheduler_mlx.py:1-100`
- **Checkpointing**: `nanochat/checkpoint_mlx.py:1-150`

**Data Loading**:
- **Dataset Loading**: `scripts/train_mlx.py:100-150`
- **Batch Creation**: `scripts/train_mlx.py:150-180`
- **SFT Data Formatting**: `scripts/sft_mlx.py:80-170`

#### ✏️ Exercises

**Exercise 4.1**: Understanding the loss function
```python
# Create: exercises/ex_4_1_loss_function.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

# Imagine we're predicting the next token
vocab_size = 10  # Simplified to 10 tokens
predicted_logits = mx.array([2.0, 1.0, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1])  # Model's prediction
true_token_id = 3  # The correct answer is token 3

print("Predicted logits:", predicted_logits.tolist())
print("True token ID:", true_token_id)

# Convert logits to probabilities
probs = nn.softmax(predicted_logits)
print("\nProbabilities (sum to 1.0):")
for i, p in enumerate(probs.tolist()):
    marker = " <-- TRUE" if i == true_token_id else ""
    print(f"  Token {i}: {p:.4f}{marker}")

# Cross-entropy loss
log_probs = nn.log_softmax(predicted_logits)
loss = -log_probs[true_token_id]
print(f"\nCross-entropy loss: {loss.item():.4f}")
print("  (Lower is better. Perfect prediction = 0)")

# What if prediction was perfect?
perfect_logits = mx.array([-100., -100., -100., 100., -100., -100., -100., -100., -100., -100.])
perfect_probs = nn.softmax(perfect_logits)
perfect_loss = -nn.log_softmax(perfect_logits)[true_token_id]
print(f"\nPerfect prediction loss: {perfect_loss.item():.4f}")

# What if prediction was terrible?
terrible_logits = mx.array([100., 100., 100., -100., 100., 100., 100., 100., 100., 100.])
terrible_probs = nn.softmax(terrible_logits)
terrible_loss = -nn.log_softmax(terrible_logits)[true_token_id]
print(f"Terrible prediction loss: {terrible_loss.item():.4f}")

# TODO: Try different logits and see how loss changes
# TODO: Understand why we use log probabilities
```

**Exercise 4.2**: Training a tiny model
```python
# Create: exercises/ex_4_2_tiny_training.py
# Train a tiny model on a tiny dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import mlx.nn as nn

# Tiny model
config = GPTConfig(sequence_len=32, vocab_size=100, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
model = GPT(config)
model.init_weights()

# Tiny dataset: just repeat the sequence [1, 2, 3, 4, 5]
train_data = mx.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=mx.int32)

# Optimizer
optimizer = AdamW(learning_rate=0.001)

# Loss function
def loss_fn(model, x, y):
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = y.reshape(-1)
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)
    return mx.mean(losses)

# Training loop
print("Training tiny model to memorize [1, 2, 3, 4, 5]...")
for step in range(100):
    # Create input/target pairs
    inputs = train_data[:, :-1]   # [1, 2, 3, 4, 5, 1, 2, 3, 4]
    targets = train_data[:, 1:]   # [2, 3, 4, 5, 1, 2, 3, 4, 5]

    # Compute loss and gradients
    loss_val, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    mx.eval(loss_val)

    if step % 10 == 0:
        print(f"Step {step:3d}: Loss = {loss_val.item():.4f}")

print("\nAfter training, let's test:")
# Test: given [1, 2], predict next tokens
test_input = mx.array([[1, 2]], dtype=mx.int32)
logits = model(test_input)
next_token_logits = logits[0, -1, :]
predicted_token = mx.argmax(next_token_logits)
print(f"Given [1, 2], model predicts: {predicted_token.item()} (should be 3)")

# TODO: Train for more steps and watch loss decrease
# TODO: Try a more complex pattern like [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# TODO: What happens if you use a lower learning rate?
```

**Exercise 4.3**: Learning rate experiments
```python
# Create: exercises/ex_4_3_learning_rate.py
# See how learning rate affects training

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import mlx.nn as nn

def train_with_lr(learning_rate, num_steps=50):
    """Train a tiny model with specified learning rate"""
    config = GPTConfig(sequence_len=32, vocab_size=100, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    model = GPT(config)
    model.init_weights()

    train_data = mx.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=mx.int32)
    optimizer = AdamW(learning_rate=learning_rate)

    def loss_fn(model, x, y):
        logits = model(x)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y.reshape(-1)
        log_probs = nn.log_softmax(logits_flat, axis=-1)
        losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)
        return mx.mean(losses)

    losses = []
    for step in range(num_steps):
        inputs = train_data[:, :-1]
        targets = train_data[:, 1:]
        loss_val, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        mx.eval(loss_val)
        losses.append(loss_val.item())

    return losses

# Try different learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1]

print("Comparing different learning rates:")
for lr in learning_rates:
    losses = train_with_lr(lr, num_steps=50)
    final_loss = losses[-1]
    print(f"\nLR = {lr:.4f}:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {losses[0] - final_loss:.4f}")

    # Check if training diverged
    if final_loss > 10:
        print(f"  ⚠️ Training diverged! LR too high!")
    elif final_loss < 0.5:
        print(f"  ✓ Good convergence!")
    else:
        print(f"  ~ Moderate convergence")

# TODO: Plot losses over time for each learning rate
# TODO: What happens with LR = 1.0? Why?
# TODO: What's the "sweet spot" learning rate?
```

**Exercise 4.4**: Understanding checkpointing
```python
# Create: exercises/ex_4_4_checkpointing.py
# Save and load model checkpoints

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.checkpoint_mlx import save_checkpoint, load_checkpoint
from nanochat.optimizers_mlx import AdamW
import mlx.core as mx
import os
import tempfile

# Train a model briefly
print("Training a tiny model...")
config = GPTConfig(sequence_len=32, vocab_size=100, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
model = GPT(config)
model.init_weights()

optimizer = AdamW(learning_rate=0.01)

# Get initial prediction
test_input = mx.array([[1, 2, 3]], dtype=mx.int32)
initial_logits = model(test_input)
initial_pred = mx.argmax(initial_logits[0, -1, :])
print(f"Initial prediction for [1,2,3]: {initial_pred.item()}")

# Train a bit (code omitted for brevity - same as previous exercises)
# ... training loop ...

# Save checkpoint
temp_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(temp_dir, "model.npz")
print(f"\nSaving checkpoint to: {checkpoint_path}")

save_checkpoint(
    model=model,
    optimizer=optimizer,
    step=100,
    loss=1.234,
    checkpoint_path=checkpoint_path,
    metadata={"note": "This is a test checkpoint"}
)

print("✓ Checkpoint saved")

# Create a NEW model (random weights)
print("\nCreating new model with random weights...")
new_model = GPT(config)
new_model.init_weights()

random_logits = new_model(test_input)
random_pred = mx.argmax(random_logits[0, -1, :])
print(f"Random model prediction: {random_pred.item()}")

# Load the checkpoint
print("\nLoading checkpoint...")
loaded_model, loaded_optimizer, metadata = load_checkpoint(
    checkpoint_path,
    model=new_model,
    optimizer=optimizer
)

print(f"✓ Loaded checkpoint from step {metadata['step']}")
print(f"  Note: {metadata['note']}")

# Test loaded model
loaded_logits = loaded_model(test_input)
loaded_pred = mx.argmax(loaded_logits[0, -1, :])
print(f"Loaded model prediction: {loaded_pred.item()}")

print(f"\nInitial == Loaded? {initial_pred.item() == loaded_pred.item()} (should be True)")

# TODO: Understand why checkpointing is crucial for training
# TODO: Think about what happens if training crashes without checkpoints
```

#### 🎯 Learning Outcomes

- Understand how models learn from data
- Know what loss functions measure
- Understand the role of optimizers and learning rates
- Appreciate the importance of checkpointing

---

### Module 5: Text Generation - Making the Model Talk

#### 🧒 ELI5 Explanation

**Imagine you're playing a word game:**
1. You start with: "Once upon a"
2. Model gives you probabilities: "time" (80%), "day" (15%), "night" (5%)
3. You pick one (maybe randomly based on probabilities)
4. Now you have: "Once upon a time"
5. Repeat!

**Generation parameters:**
- **Temperature**: How random should we be?
  - Low (0.1): Always pick the best word (boring but safe)
  - High (1.5): Pick random words (creative but risky)
- **Top-K**: Only consider the top K most likely tokens
- **Repetition penalty**: Discourage repeating the same words

#### 📖 Concepts Covered

1. **Greedy Sampling**: Always pick the most likely token
2. **Temperature Sampling**: Add randomness
3. **Top-K Sampling**: Limit choices to top K tokens
4. **Top-P (Nucleus) Sampling**: Limit choices by probability mass
5. **Repetition Penalty**: Reduce probability of recently used tokens
6. **Stopping Criteria**: When to stop generating
7. **KV Cache**: Speed up generation

#### 🔍 Code References

- **Generation Loop**: `scripts/chat_cli_mlx.py:46-136`
- **Temperature Sampling**: `scripts/chat_cli_mlx.py:103-105`
- **Top-K Filtering**: `scripts/chat_cli_mlx.py:108-116`
- **Repetition Penalty**: `scripts/chat_cli_mlx.py:88-101`
- **Token Sampling**: `scripts/chat_cli_mlx.py:119`

#### ✏️ Exercises

**Exercise 5.1**: Temperature experiments
```python
# Create: exercises/ex_5_1_temperature.py
# See how temperature affects generation

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

# Simulate a model's prediction
vocab_size = 10
logits = mx.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

print("Original logits:", logits.tolist())
print("\nSampling with different temperatures:\n")

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temperatures:
    # Apply temperature
    scaled_logits = logits / temp
    probs = nn.softmax(scaled_logits)

    print(f"Temperature = {temp}:")
    print(f"  Top 5 probabilities: {probs[:5].tolist()}")

    # Sample 10 times
    samples = []
    for _ in range(10):
        token = mx.random.categorical(scaled_logits)
        samples.append(token.item())

    print(f"  10 samples: {samples}")
    print(f"  Unique tokens: {len(set(samples))}/10")
    print()

# TODO: Low temperature = deterministic (always same token)
# TODO: High temperature = random (many different tokens)
# TODO: Find the right balance for creative but coherent text
```

**Exercise 5.2**: Implementing basic generation
```python
# Create: exercises/ex_5_2_basic_generation.py
# Build a simple text generator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
import mlx.core as mx
import mlx.nn as nn
import os

# Load tokenizer
tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# Small untrained model (for demonstration)
config = GPTConfig(sequence_len=256, vocab_size=65536, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)
model = GPT(config)
model.init_weights()

def generate(prompt_text, max_tokens=20, temperature=1.0):
    """Simple generation function"""
    # Encode prompt
    tokens = tokenizer.encode(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Encoded: {tokens}\n")
    print("Generated: ", end="")

    for _ in range(max_tokens):
        # Get model predictions
        input_array = mx.array([tokens], dtype=mx.int32)
        logits = model(input_array)

        # Get logits for last position
        next_logits = logits[0, -1, :] / temperature

        # Sample
        next_token = mx.random.categorical(next_logits)
        next_token_id = int(next_token.item())

        # Decode and print
        token_text = tokenizer.decode([next_token_id])
        print(token_text, end="", flush=True)

        # Add to sequence
        tokens.append(next_token_id)

    print("\n")

# Try it!
generate("Once upon a time", max_tokens=30, temperature=1.0)

# TODO: This will be gibberish because model is untrained!
# TODO: Try with a trained checkpoint to see real generation
# TODO: Add top-k filtering
# TODO: Add repetition penalty
```

**Exercise 5.3**: Comparing sampling strategies
```bash
# Create: exercises/ex_5_3_sampling_comparison.sh
# Compare different sampling strategies

# You'll need a trained checkpoint for this!
# Assuming you have checkpoints/d10_sft/best.npz

echo "=== Greedy (temperature=0.1, top_k=1) ==="
python scripts/chat_cli_mlx.py \
  --checkpoint checkpoints/d10_sft/best.npz \
  --prompt "What is the capital of France?" \
  --temperature 0.1 \
  --top-k 1 \
  --max-tokens 50

echo -e "\n=== Moderate randomness (temperature=0.8, top_k=50) ==="
python scripts/chat_cli_mlx.py \
  --checkpoint checkpoints/d10_sft/best.npz \
  --prompt "What is the capital of France?" \
  --temperature 0.8 \
  --top-k 50 \
  --max-tokens 50

echo -e "\n=== High creativity (temperature=1.5, top_k=100) ==="
python scripts/chat_cli_mlx.py \
  --checkpoint checkpoints/d10_sft/best.npz \
  --prompt "What is the capital of France?" \
  --temperature 1.5 \
  --top-k 100 \
  --max-tokens 50

# TODO: Run this and compare outputs
# TODO: Which is most accurate?
# TODO: Which is most creative?
# TODO: When would you use each strategy?
```

**Exercise 5.4**: Understanding KV cache speedup
```python
# Create: exercises/ex_5_4_kv_cache_speed.py
# Measure KV cache performance impact

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.kv_cache_mlx import KVCache
import mlx.core as mx
import time

config = GPTConfig(sequence_len=512, vocab_size=65536, n_layer=6, n_head=6, n_kv_head=6, n_embd=384)
model = GPT(config)
model.init_weights()

def generate_no_cache(prompt_tokens, num_tokens):
    """Generate without KV cache (slow)"""
    tokens = list(prompt_tokens)

    for _ in range(num_tokens):
        # Reprocess entire sequence each time!
        input_array = mx.array([tokens], dtype=mx.int32)
        logits = model(input_array)
        mx.eval(logits)

        next_token = mx.argmax(logits[0, -1, :])
        tokens.append(int(next_token.item()))

    return tokens

def generate_with_cache(prompt_tokens, num_tokens):
    """Generate with KV cache (fast)"""
    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer
    )

    # Process prompt once
    prompt_array = mx.array([list(prompt_tokens)], dtype=mx.int32)
    logits = model(prompt_array, kv_cache=kv_cache)
    mx.eval(logits)

    tokens = list(prompt_tokens)

    # Generate tokens one at a time
    for _ in range(num_tokens):
        next_token = mx.argmax(logits[0, -1, :])
        next_token_id = int(next_token.item())
        tokens.append(next_token_id)

        # Only process the new token!
        next_input = mx.array([[next_token_id]], dtype=mx.int32)
        logits = model(next_input, kv_cache=kv_cache)
        mx.eval(logits)

    return tokens

# Benchmark
prompt = [1, 2, 3, 4, 5]  # Simple prompt
num_generate = 50

print("Benchmarking generation speed...\n")

# Without cache
print("Without KV cache:")
start = time.time()
result_no_cache = generate_no_cache(prompt, num_generate)
time_no_cache = time.time() - start
print(f"  Generated {num_generate} tokens in {time_no_cache:.3f}s")
print(f"  Speed: {num_generate/time_no_cache:.1f} tokens/sec")

# With cache
print("\nWith KV cache:")
start = time.time()
result_with_cache = generate_with_cache(prompt, num_generate)
time_with_cache = time.time() - start
print(f"  Generated {num_generate} tokens in {time_with_cache:.3f}s")
print(f"  Speed: {num_generate/time_with_cache:.1f} tokens/sec")

print(f"\n⚡ Speedup: {time_no_cache/time_with_cache:.1f}x faster!")

# TODO: Try generating 100 or 200 tokens
# TODO: Notice how speedup increases with more tokens generated
# TODO: Understand why: without cache, time is O(n²), with cache is O(n)
```

#### 🎯 Learning Outcomes

- Understand different sampling strategies
- Know when to use greedy vs. creative sampling
- Appreciate the massive speedup from KV cache
- Be able to tune generation parameters

---

### Module 6: The Training Pipeline

#### 🧒 ELI5 Explanation

**Training an LLM is like baking a multi-layer cake:**

**Stage 1 - Pretraining (The base cake):**
- Use ALL the text from the internet
- Teach the model: "predict the next word"
- Takes a lot of compute (days/weeks)
- Result: Model knows language, facts, reasoning

**Stage 2 - SFT (The frosting):**
- Use instruction-following examples
- Teach the model: "answer questions helpfully"
- Takes much less time (hours)
- Result: Model follows instructions, acts like a helpful assistant

**Stage 3 - Inference (Serving the cake):**
- People use your model
- Fast generation (KV cache!)
- Maybe deployed on servers

#### 📖 Concepts Covered

1. **Pretraining**: Learning language from raw text
2. **Supervised Fine-Tuning (SFT)**: Learning to follow instructions
3. **Data Preparation**: How to prepare training data
4. **Training Hyperparameters**: Batch size, learning rate, etc.
5. **Evaluation Metrics**: Loss, perplexity, accuracy
6. **Checkpointing Strategy**: When to save
7. **Distributed Training**: Using multiple GPUs (PyTorch version)

#### 🔍 Code References

**Pretraining**:
- **Main Script**: `scripts/train_mlx.py:1-400`
- **Data Loading**: `scripts/train_mlx.py:100-180`
- **Training Loop**: `scripts/train_mlx.py:200-350`
- **Configuration**: `scripts/train_mlx.py:20-80`

**SFT**:
- **Main Script**: `scripts/sft_mlx.py:1-462`
- **Instruction Formatting**: `scripts/sft_mlx.py:80-116`
- **Masked Loss**: `scripts/sft_mlx.py:173-200`
- **Training Loop**: `scripts/sft_mlx.py:300-401`

**Common Components**:
- **Checkpointing**: `nanochat/checkpoint_mlx.py:1-150`
- **Optimizers**: `nanochat/optimizers_mlx.py:1-100`
- **LR Scheduling**: `nanochat/lr_scheduler_mlx.py:1-100`

#### ✏️ Exercises

**Exercise 6.1**: Analyze a training run
```bash
# Create: exercises/ex_6_1_analyze_training.sh
# Look at training logs to understand the process

# If you have a training log:
cat sft_training.log | grep "Step.*Loss"

# Or run a short training:
python scripts/sft_mlx.py \
  --base-checkpoint checkpoints/d10_pretrain/best.npz \
  --output-dir exercises/test_sft \
  --batch-size 2 \
  --max-steps 50 \
  --log-interval 5

# TODO: Watch the loss decrease over time
# TODO: Notice learning rate changing
# TODO: See validation runs
```

**Exercise 6.2**: Understanding masked loss in SFT
```python
# Create: exercises/ex_6_2_masked_loss.py
# See why we mask user prompts in SFT

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer
import os

tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# Example conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}

tokens, mask = tokenizer.render_conversation(conversation)

print("Tokenized conversation:")
for i, (token_id, mask_val) in enumerate(zip(tokens, mask)):
    token_text = tokenizer.decode([token_id])
    mask_indicator = "✓ TRAIN" if mask_val == 1 else "✗ SKIP"
    print(f"  {i:3d}: {token_id:6d} '{token_text:20s}' {mask_indicator}")

print("\nKey insight:")
print("  - User message tokens: SKIP (mask=0)")
print("  - Assistant message tokens: TRAIN (mask=1)")
print("\nWhy?")
print("  - We want the model to GENERATE assistant responses")
print("  - We DON'T train it to generate user messages")
print("  - Loss is only computed on assistant tokens!")

# TODO: Create a different conversation and see the mask
# TODO: Understand why this makes models better at being assistants
```

**Exercise 6.3**: Hyperparameter tuning
```python
# Create: exercises/ex_6_3_hyperparameters.py
# Document showing hyperparameter choices

# This is a learning exercise - read and understand!

HYPERPARAMETERS = {
    "pretraining": {
        "batch_size": 16,  # How many examples per update
        "learning_rate": 1e-3,  # How big each update is
        "max_steps": 10000,  # How long to train
        "sequence_len": 1024,  # Max context length
        "warmup_steps": 500,  # LR warmup period

        "why": {
            "batch_size": "Larger = more stable gradients, but uses more memory",
            "learning_rate": "Too high = unstable, too low = slow learning",
            "max_steps": "More steps = better model, but takes longer",
            "sequence_len": "Longer = more context, but slower training",
            "warmup_steps": "Prevents instability at the start",
        }
    },

    "sft": {
        "batch_size": 4,  # Smaller because sequences vary in length
        "learning_rate": 5e-5,  # Much lower than pretraining!
        "max_steps": 3000,  # Less than pretraining
        "sequence_len": 512,  # Shorter than pretraining

        "why": {
            "batch_size": "Smaller for memory efficiency with padding",
            "learning_rate": "Low to avoid catastrophic forgetting",
            "max_steps": "Fine-tuning is faster than pretraining",
            "sequence_len": "Instructions are usually shorter",
        }
    },

    "model_architecture": {
        "n_layer": 10,  # Number of transformer layers
        "n_head": 8,  # Number of attention heads
        "n_embd": 512,  # Embedding dimension
        "vocab_size": 65536,  # Size of tokenizer vocabulary

        "why": {
            "n_layer": "More layers = more powerful, but slower",
            "n_head": "More heads = attend to more patterns",
            "n_embd": "Larger = more capacity, but more parameters",
            "vocab_size": "Determined by tokenizer (fixed)",
        }
    }
}

# Print the guide
for stage, params in HYPERPARAMETERS.items():
    print(f"\n{'='*60}")
    print(f"{stage.upper()} HYPERPARAMETERS")
    print(f"{'='*60}")

    why_dict = params.pop("why", {})

    for param, value in params.items():
        print(f"\n{param}: {value}")
        if param in why_dict:
            print(f"  → {why_dict[param]}")

# TODO: Try changing hyperparameters in a real training run
# TODO: Document what happens when you change each one
```

**Exercise 6.4**: Checkpoint analysis
```python
# Create: exercises/ex_6_4_checkpoint_analysis.py
# Inspect checkpoint metadata

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os

# Look at checkpoint metadata
checkpoint_dirs = [
    "checkpoints/d10_pretrain_causal_fix",
    "checkpoints/d10_sft",
]

for ckpt_dir in checkpoint_dirs:
    if not os.path.exists(ckpt_dir):
        continue

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    # Find best checkpoint
    best_path = os.path.join(ckpt_dir, "best.npz.meta.json")
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            meta = json.load(f)

        print(f"\nMetadata:")
        print(f"  Step: {meta.get('step', 'N/A')}")
        print(f"  Loss: {meta.get('loss', 'N/A'):.4f}")

        if 'model_config' in meta:
            config = meta['model_config']
            print(f"\nModel Config:")
            print(f"  Layers: {config.get('n_layer')}")
            print(f"  Heads: {config.get('n_head')}")
            print(f"  Embedding dim: {config.get('n_embd')}")
            print(f"  Vocab size: {config.get('vocab_size')}")

        # Calculate model size
        # Rough estimate: 2 bytes per parameter (bfloat16)
        if 'model_config' in meta:
            n_layer = config.get('n_layer', 0)
            n_embd = config.get('n_embd', 0)
            vocab_size = config.get('vocab_size', 0)

            # Simplified parameter count
            # Embeddings: vocab_size * n_embd * 2 (token + position)
            # Transformer: ~12 * n_layer * n_embd^2
            # Output: vocab_size * n_embd

            param_count = (
                2 * vocab_size * n_embd +
                12 * n_layer * n_embd * n_embd +
                vocab_size * n_embd
            )

            size_mb = param_count * 2 / (1024 * 1024)  # 2 bytes per param

            print(f"\nEstimated:")
            print(f"  Parameters: ~{param_count/1e6:.1f}M")
            print(f"  Size: ~{size_mb:.1f} MB")

# TODO: Compare checkpoint sizes
# TODO: Understand relationship between architecture and size
```

#### 🎯 Learning Outcomes

- Understand the complete training pipeline
- Know the difference between pretraining and SFT
- Understand why hyperparameters matter
- Be able to analyze training runs

---

### Module 7: Putting It All Together

#### 🧒 ELI5 Explanation

**You've learned all the pieces, now let's see the full picture:**

1. **Text** → Tokenizer → **Numbers**
2. **Numbers** → Model → **Predictions**
3. **Predictions** → Sampling → **Generated text**

**Training cycle:**
1. Get text data
2. Tokenize it
3. Feed to model
4. Compute loss (how wrong?)
5. Update weights
6. Repeat millions of times

**Using the model:**
1. User types prompt
2. Tokenize
3. Model predicts next token
4. Sample from predictions
5. Repeat until done
6. Detokenize back to text

#### 📖 Concepts Covered

1. **End-to-End Pipeline**: From raw text to trained model
2. **Production Deployment**: How to serve models
3. **Evaluation**: Measuring model quality
4. **Debugging**: What to do when things go wrong
5. **Scaling**: Making models bigger and better

#### ✏️ Exercises

**Exercise 7.1**: Train a model from scratch (mini version)
```bash
# Create: exercises/ex_7_1_train_from_scratch.sh
# Train a tiny model on a small dataset

# This will take ~10 minutes on a Mac

python scripts/train_mlx.py \
  --model-size d2 \
  --batch-size 8 \
  --max-steps 1000 \
  --learning-rate 0.001 \
  --output-dir exercises/my_first_model \
  --log-interval 50

# Then test it:
python scripts/chat_cli_mlx.py \
  --checkpoint exercises/my_first_model/best.npz \
  --mode plain \
  --prompt "Once upon a time"

# TODO: Watch the training process
# TODO: See loss decrease
# TODO: Test the model (it will be basic but it's YOURS!)
```

**Exercise 7.2**: Compare model sizes
```python
# Create: exercises/ex_7_2_model_sizes.py
# Compare different model sizes

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
import time
import mlx.core as mx

def benchmark_model(config_name, config):
    """Benchmark model creation and inference"""
    print(f"\n{'='*60}")
    print(f"Model: {config_name}")
    print(f"{'='*60}")

    # Create model
    model = GPT(config)
    model.init_weights()

    # Count parameters
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

    nparams = count_params(model.parameters())

    print(f"Config:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding: {config.n_embd}")
    print(f"  Parameters: {nparams:,} ({nparams/1e6:.2f}M)")

    # Benchmark inference
    batch_size = 1
    seq_len = 128
    dummy_input = mx.ones((batch_size, seq_len), dtype=mx.int32)

    # Warmup
    for _ in range(3):
        _ = model(dummy_input)
        mx.eval(model.parameters())

    # Benchmark
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        logits = model(dummy_input)
        mx.eval(logits)
    elapsed = time.time() - start

    time_per_run = elapsed / num_runs
    tokens_per_sec = seq_len / time_per_run

    print(f"\nInference Speed:")
    print(f"  Time per forward pass: {time_per_run*1000:.1f}ms")
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")

    return nparams, time_per_run

# Different model sizes
configs = {
    "d2-tiny": GPTConfig(sequence_len=256, vocab_size=65536, n_layer=2, n_head=4, n_kv_head=4, n_embd=128),
    "d6-small": GPTConfig(sequence_len=512, vocab_size=65536, n_layer=6, n_head=6, n_kv_head=6, n_embd=384),
    "d10-medium": GPTConfig(sequence_len=1024, vocab_size=65536, n_layer=10, n_head=8, n_kv_head=8, n_embd=512),
    "d14-large": GPTConfig(sequence_len=1024, vocab_size=65536, n_layer=14, n_head=12, n_kv_head=12, n_embd=768),
}

results = {}
for name, config in configs.items():
    params, speed = benchmark_model(name, config)
    results[name] = (params, speed)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, (params, speed) in results.items():
    print(f"{name:15s}: {params/1e6:6.1f}M params, {speed*1000:6.1f}ms per forward pass")

# TODO: See the tradeoff between size and speed
# TODO: Which model would you choose for your use case?
```

**Exercise 7.3**: Evaluate model quality
```python
# Create: exercises/ex_7_3_evaluation.py
# Create a simple evaluation benchmark

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.checkpoint_mlx import load_checkpoint
import mlx.core as mx
import mlx.nn as nn
import os
import json

# Evaluation dataset
eval_data = [
    {
        "prompt": "The capital of France is",
        "expected_word": "Paris",
        "category": "factual"
    },
    {
        "prompt": "2 + 2 =",
        "expected_word": "4",
        "category": "math"
    },
    {
        "prompt": "The Earth orbits the",
        "expected_word": "Sun",
        "category": "science"
    },
    {
        "prompt": "The color of the sky is",
        "expected_word": "blue",
        "category": "common_sense"
    },
]

def evaluate_model(checkpoint_path):
    """Evaluate model on simple prompts"""
    print(f"Evaluating: {checkpoint_path}\n")

    # Load tokenizer
    tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    # Load model
    meta_path = checkpoint_path + ".meta.json"
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    config = GPTConfig(**metadata['model_config'])
    model = GPT(config)
    model, _, _ = load_checkpoint(checkpoint_path, model=model)

    # Evaluate
    results = []
    for example in eval_data:
        prompt = example["prompt"]
        expected = example["expected_word"]

        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        input_array = mx.array([tokens], dtype=mx.int32)

        # Get predictions
        logits = model(input_array)
        next_token_logits = logits[0, -1, :]

        # Get top 5 predictions
        top_5_ids = mx.argpartition(-next_token_logits, 5)[:5].tolist()
        top_5_tokens = [tokenizer.decode([tid]) for tid in top_5_ids]

        # Check if expected is in top 5
        expected_tokens = tokenizer.encode(expected)
        if expected_tokens:
            expected_id = expected_tokens[0]
            rank = None
            for i, tid in enumerate(top_5_ids):
                if tid == expected_id:
                    rank = i + 1
                    break

            success = rank is not None
        else:
            success = False
            rank = None

        results.append({
            "prompt": prompt,
            "expected": expected,
            "top_5": top_5_tokens,
            "success": success,
            "rank": rank,
        })

        status = f"✓ (rank {rank})" if success else "✗"
        print(f"{status} '{prompt}' → {top_5_tokens[0]} (expected: {expected})")

    # Summary
    accuracy = sum(r["success"] for r in results) / len(results)
    print(f"\nAccuracy: {accuracy*100:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")

    return results

# Evaluate your checkpoints
if os.path.exists("checkpoints/d10_sft/best.npz"):
    evaluate_model("checkpoints/d10_sft/best.npz")

# TODO: Create more evaluation examples
# TODO: Compare pretrained vs SFT model
# TODO: Track accuracy across training checkpoints
```

**Exercise 7.4**: Build a simple chatbot
```python
# Create: exercises/ex_7_4_simple_chatbot.py
# Build a minimal chatbot using what you've learned

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt_mlx import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.kv_cache_mlx import KVCache
from nanochat.checkpoint_mlx import load_checkpoint
import mlx.core as mx
import os
import json

class SimpleChatbot:
    """A minimal chatbot implementation"""

    def __init__(self, checkpoint_path):
        # Load tokenizer
        tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
        self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        # Load model
        meta_path = checkpoint_path + ".meta.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        config = GPTConfig(**metadata['model_config'])
        self.model = GPT(config)
        self.model, _, _ = load_checkpoint(checkpoint_path, model=self.model)

        # Setup KV cache
        self.kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.sequence_len * 10,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer
        )

        # Special tokens
        self.bos = self.tokenizer.get_bos_token_id()
        self.user_start = self.tokenizer.encode_special("<|user_start|>")
        self.user_end = self.tokenizer.encode_special("<|user_end|>")
        self.assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        # Conversation state
        self.conversation_tokens = [self.bos]

    def chat(self, user_message, max_tokens=256, temperature=0.8):
        """Send a message and get a response"""
        # Add user message
        self.conversation_tokens.append(self.user_start)
        self.conversation_tokens.extend(self.tokenizer.encode(user_message))
        self.conversation_tokens.append(self.user_end)
        self.conversation_tokens.append(self.assistant_start)

        # Generate response
        response_tokens = []
        tokens_array = mx.array([self.conversation_tokens], dtype=mx.int32)
        logits = self.model(tokens_array, kv_cache=self.kv_cache)
        mx.eval(logits)

        for _ in range(max_tokens):
            next_token_logits = logits[0, -1, :] / temperature
            next_token = mx.random.categorical(next_token_logits)
            next_token_id = int(next_token.item())

            # Stop if we hit assistant_end
            if next_token_id == self.assistant_end:
                break

            response_tokens.append(next_token_id)

            # Process next token
            next_input = mx.array([[next_token_id]], dtype=mx.int32)
            logits = self.model(next_input, kv_cache=self.kv_cache)
            mx.eval(logits)

        # Add to conversation
        response_tokens.append(self.assistant_end)
        self.conversation_tokens.extend(response_tokens)

        # Decode response
        response_text = self.tokenizer.decode(response_tokens[:-1])  # Exclude end token
        return response_text

    def reset(self):
        """Start a new conversation"""
        self.conversation_tokens = [self.bos]
        self.kv_cache.reset()

# Use it!
if os.path.exists("checkpoints/d10_sft/best.npz"):
    print("Loading chatbot...")
    bot = SimpleChatbot("checkpoints/d10_sft/best.npz")
    print("Ready! Type 'quit' to exit, 'reset' to start new conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            bot.reset()
            print("(Conversation reset)\n")
            continue

        if user_input:
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")

# TODO: Add more features (memory, personality, tools)
# TODO: Experiment with different prompting strategies
# TODO: Build a web UI for your chatbot
```

#### 🎯 Learning Outcomes

- Understand the complete pipeline
- Be able to train and evaluate models
- Know how to use models in applications
- Have built a working chatbot!

---

## 🎓 Final Project Ideas

After completing the course, try these projects:

### Project 1: Domain-Specific Fine-Tuning
- Pick a domain (e.g., cooking, programming, history)
- Collect instruction data for that domain
- Fine-tune a model
- Evaluate on domain-specific tasks

### Project 2: Multi-Turn Conversation System
- Build a chatbot that remembers context across multiple turns
- Implement conversation summarization
- Add personality and style

### Project 3: Model Compression
- Take a trained model
- Implement quantization (reduce precision)
- Measure size vs. quality tradeoffs

### Project 4: Custom Tokenizer
- Train a BPE tokenizer on a new language or domain
- Replace the existing tokenizer
- Retrain the model with your tokenizer

### Project 5: Evaluation Benchmark
- Create a comprehensive evaluation suite
- Test models on factual knowledge, reasoning, creativity
- Compare different model sizes and training strategies

---

## 📖 Additional Resources

### Code References Quick Index

**Core Architecture**:
- `nanochat/gpt_mlx.py` - GPT model
- `nanochat/kv_cache_mlx.py` - KV cache optimization
- `nanochat/tokenizer.py` - Tokenization

**Training**:
- `scripts/train_mlx.py` - Pretraining
- `scripts/sft_mlx.py` - Supervised fine-tuning
- `nanochat/optimizers_mlx.py` - AdamW optimizer
- `nanochat/lr_scheduler_mlx.py` - Learning rate scheduling

**Inference**:
- `scripts/chat_cli_mlx.py` - Interactive chat
- `nanochat/checkpoint_mlx.py` - Checkpoint management

**PyTorch Equivalents** (for reference):
- `scripts/pretrain.py` - PyTorch pretraining
- `scripts/mid_train.py` - PyTorch mid-training
- `scripts/chat_sft.py` - PyTorch SFT

### Key Papers to Read

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - The original Transformer paper

2. **"Language Models are Few-Shot Learners"** (GPT-3, Brown et al., 2020)
   - Large-scale language model training

3. **"Training language models to follow instructions"** (InstructGPT, Ouyang et al., 2022)
   - Supervised fine-tuning and RLHF

### Recommended Learning Path

**Week 1**: Modules 1-2 (Overview + Tokenization)
**Week 2**: Modules 3-4 (Architecture + Training)
**Week 3**: Modules 5-6 (Generation + Pipeline)
**Week 4**: Module 7 + Final Project

---

## 🎯 Learning Outcomes Checklist

By the end of this course, you should be able to:

- [ ] Explain how LLMs work to a non-technical person
- [ ] Understand every component of the Transformer architecture
- [ ] Read and modify the nanochat codebase
- [ ] Train a model from scratch (with pretraining)
- [ ] Fine-tune a model for specific tasks
- [ ] Implement different generation strategies
- [ ] Debug training issues
- [ ] Evaluate model quality
- [ ] Build a working application using an LLM
- [ ] Understand the tradeoffs in model design

---

## 🚀 Next Steps After This Course

1. **Dive Deeper**:
   - Implement advanced techniques (RLHF, DPO)
   - Explore different architectures (Llama, Mistral)
   - Study model compression and quantization

2. **Build Real Applications**:
   - Create a specialized chatbot
   - Build a code completion tool
   - Develop a creative writing assistant

3. **Contribute**:
   - Contribute to open-source LLM projects
   - Share your learnings with others
   - Build and release your own models

4. **Keep Learning**:
   - Follow the latest research
   - Experiment with new techniques
   - Join the LLM community

---

## 📝 Notes for Course Developers

When creating the full course from this plan:

1. **Add Visuals**: Diagrams for architecture, data flow, attention mechanism
2. **Video Walkthroughs**: Record yourself running the exercises
3. **Interactive Notebooks**: Convert exercises to Jupyter notebooks
4. **Automated Grading**: Create tests for exercises
5. **Discussion Forums**: Set up Q&A for students
6. **Office Hours**: Live sessions for difficult topics
7. **Capstone Project**: Guided final project with milestones

---

**Good luck on your LLM learning journey! 🚀**
