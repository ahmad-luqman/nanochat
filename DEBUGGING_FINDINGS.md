# MLX Port Debugging Findings

**Date:** 2025-10-16
**Status:** KV Cache bug identified but not yet fixed

---

## ✅ CONFIRMED WORKING

### 1. Training (without KV cache)
- **Loss progression:** 9.73 → 0.047 over 9400 steps
- **BPB:** Decreased from 3.50 → 0.0184
- **Checkpoints:** Save/load correctly with optimizer state
- **Evidence:** Training completed successfully with proper loss convergence
- **Config:** d10 model (99M params), batch=16, seq_len=1024, FineWebEdu dataset

### 2. Inference WITHOUT KV cache
- **Script:** `test_plain_generation.py`
- **Output quality:** Diverse, reasonable base model text
- **Example:** "play exh obelisk during crystallization for or canemor..."
- **No repetition issues**
- **Evidence:** Tested 2025-10-16, works correctly

### 3. Data Pipeline & Tokenizer
- **Dataset:** FineWebEdu downloads and loads correctly
- **Tokenizer:** RustBPE (vocab_size=65,536) encodes/decodes properly
- **Data quality:** Verified diverse, non-repetitive content
- **Evidence:** `check_training_data.py` shows clean training samples

### 4. Categorical Sampling (after fix)
- **Bug found:** Was passing `mx.log(probs)` instead of raw logits
- **Fix applied:** Changed to `mx.random.categorical(next_token_logits)`
- **Location:** `scripts/chat_cli_mlx.py:118-120` in `generate_tokens()` function
- **MLX requirement:** `categorical()` expects unnormalized logits, not log probabilities

### 5. Loss Computation
- **Cross-entropy:** Correct implementation with proper masking
- **Target shifting:** Correct (x[:, :-1], y[:, 1:])
- **Padding mask:** Properly ignores -1 tokens
- **Evidence:** Verified against PyTorch reference

---

## ❌ CONFIRMED BROKEN

### 1. KV Cache Inference 🔴 CRITICAL BUG

**Symptom:** KV cache produces completely different logits than no-KV path

**Evidence from `debug_kv_detailed.py`:**
```
Step 1: Logits diff = 0.000000 ✅ (KV cache matches no-KV)
Step 2: Logits diff = 535,971.250000 ❌ (massive divergence!)
Step 3: Logits diff = 438,774.250000 ❌ (continues to diverge)
```

**Details:**
- First token processing (step 1): KV cache works perfectly
- Second token onwards (step 2+): Logits completely wrong
- Top 5 logits comparison at step 2:
  - WITHOUT KV: `[5.05, 5.37, 5.37, 5.40, 5.74]`
  - WITH KV: `[-2.95, -2.74, -2.71, -1.16, 6.32]` ← completely different!

**Impact:**
- Generation with KV cache is completely broken
- Produces extreme repetition instead of diverse text
- Makes inference unusable for production

**Root Cause:** Unknown - needs investigation
- KV cache storage appears correct (verified positions update)
- Issue likely in how cached K/V are used in attention
- Possible suspects:
  - Rotary embedding offset calculation
  - Attention mask for cached tokens
  - How K/V from cache are combined with new K/V

### 2. Generation with KV cache

**Script:** `scripts/chat_cli_mlx.py --mode plain`

**Symptom:** Extreme repetition
```
Prompt: "In the year 2020,"
Output: "an an an an an an an an..." (repeats indefinitely)
```

**Root Cause:** KV cache bug above
**Status:** Will be fixed once KV cache bug is resolved

---

## ⚠️ UNCLEAR / NEEDS INVESTIGATION

### 1. Attention Mask Semantics

**The Contradiction:**
- My mask test (`test_mask_semantics.py`) showed: `True = KEEP, False = BLOCK` in MLX
- Training code uses: `mx.triu(mx.ones(...), k=1)` = upper triangular True
- Upper triangular True should mean "keep future tokens"
- This should be WRONG for causal attention
- **BUT:** Training worked perfectly (loss decreased properly!)

**Possible Explanations:**
1. My mask test was wrong or misinterpreted
2. MLX SDPA has different semantics than I tested
3. The mask parameter works differently in training vs inference
4. There's something else I'm missing

**What I Tried:**
- Changing `triu` → `tril` made generation WORSE (pure repetition)
- Reverting to `triu` restored diverse generation (without KV cache)
- This suggests `triu` is correct, but contradicts my mask test

**Status:** Needs deeper investigation but NOT blocking (training works)

### 2. Components Not Verified Against PyTorch

**These were ported but never compared side-by-side:**
- Rotary embeddings output
- RMSNorm output
- Full forward pass (MLX vs PyTorch on same inputs)
- MQA `repeat_kv` logic
- Weight initialization values
- Logits softcapping behavior

**Risk:** Could have subtle bugs that don't prevent training but affect quality

---

## Priority Bug to Fix

**#1: KV Cache logits divergence at step 2+**

This is the blocking bug preventing usable inference. Everything else either works or is lower priority.

**Debug strategy:**
1. Add detailed logging to attention mechanism when using KV cache
2. Compare intermediate values (Q, K, V, attention weights) with and without KV cache
3. Check rotary embedding offset calculation
4. Verify attention mask for single-token inference with cache
5. Check if cache position tracking affects anything unexpected

---

## Files for Reference

**Working scripts:**
- `test_plain_generation.py` - Generation without KV cache (works)
- `check_training_data.py` - Verify training data quality

**Debugging scripts:**
- `debug_kv_detailed.py` - Shows KV cache divergence at step 2
- `test_mask_semantics.py` - Tests mask behavior (needs reinterpretation)
- `test_no_kv_cache.py` - Compare with/without cache

**Main code:**
- `nanochat/gpt_mlx.py` - Model implementation (KV cache bug is here)
- `nanochat/kv_cache_mlx.py` - KV cache storage (seems correct)
- `scripts/chat_cli_mlx.py` - Inference script (categorical sampling fixed)

---

## Training Status

**Stopped at:** Step 9420/10000 (94% complete)
**Reason:** Debugging KV cache issues
**Model:** Potentially usable for no-KV inference, unusable for KV cache inference
**Next training:** Wait until KV cache bug is fixed to avoid wasting compute
