# Course Plan Implementation Tasks

This document tracks the implementation progress of the LLM learning course.

## Status Legend
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- 🧪 Needs Testing

---

## Phase 1: Module 1 - The Big Picture

### Exercise 1.1: Run a pretrained model ⬜
- **File**: `exercises/ex_1_1_run_model.sh`
- **Description**: Shell script to run model with different prompts
- **Dependencies**: Trained checkpoint
- **Test**: Run and verify output

### Exercise 1.2: Compare base vs SFT ⬜
- **File**: `exercises/ex_1_2_compare_models.sh`
- **Description**: Shell script comparing base and SFT models
- **Dependencies**: Both checkpoints
- **Test**: Run and verify comparison shows clear differences

---

## Phase 2: Module 2 - Tokenization

### Exercise 2.1: Tokenizer basics ⬜
- **File**: `exercises/ex_2_1_tokenizer_basics.py`
- **Description**: Explore encoding/decoding
- **Dependencies**: Tokenizer downloaded
- **Test**: Run and verify tokenization works

### Exercise 2.2: Special tokens ⬜
- **File**: `exercises/ex_2_2_special_tokens.py`
- **Description**: Understand special tokens and conversation formatting
- **Dependencies**: Tokenizer
- **Test**: Verify conversation rendering

### Exercise 2.3: Token efficiency ⬜
- **File**: `exercises/ex_2_3_token_efficiency.py`
- **Description**: Compare tokenization of different texts
- **Dependencies**: Tokenizer
- **Test**: Verify ratios calculated correctly

---

## Phase 3: Module 3 - Architecture

### Exercise 3.1: Model architecture ⬜
- **File**: `exercises/ex_3_1_model_architecture.py`
- **Description**: Create models of different sizes, count parameters
- **Dependencies**: MLX, nanochat.gpt_mlx
- **Test**: Verify parameter counts

### Exercise 3.2: Forward pass ⬜
- **File**: `exercises/ex_3_2_forward_pass.py`
- **Description**: Watch data flow through model
- **Dependencies**: MLX, model
- **Test**: Verify shapes are correct

### Exercise 3.3: Attention (conceptual) ⬜
- **File**: `exercises/ex_3_3_attention.py`
- **Description**: Conceptual understanding of attention
- **Dependencies**: None (prints explanations)
- **Test**: Read and verify explanations are clear

### Exercise 3.4: KV cache ⬜
- **File**: `exercises/ex_3_4_kv_cache.py`
- **Description**: Benchmark generation with/without KV cache
- **Dependencies**: MLX, model, KV cache
- **Test**: Verify speedup is significant

---

## Phase 4: Module 4 - Training

### Exercise 4.1: Loss function ⬜
- **File**: `exercises/ex_4_1_loss_function.py`
- **Description**: Understand cross-entropy loss
- **Dependencies**: MLX
- **Test**: Verify loss calculations

### Exercise 4.2: Tiny training ⬜
- **File**: `exercises/ex_4_2_tiny_training.py`
- **Description**: Train tiny model on simple sequence
- **Dependencies**: MLX, model, optimizer
- **Test**: Verify loss decreases

### Exercise 4.3: Learning rate ⬜
- **File**: `exercises/ex_4_3_learning_rate.py`
- **Description**: Compare different learning rates
- **Dependencies**: MLX, model, optimizer
- **Test**: Verify LR affects convergence

### Exercise 4.4: Checkpointing ⬜
- **File**: `exercises/ex_4_4_checkpointing.py`
- **Description**: Save and load checkpoints
- **Dependencies**: Checkpoint manager
- **Test**: Verify save/load roundtrip works

---

## Phase 5: Module 5 - Generation

### Exercise 5.1: Temperature ⬜
- **File**: `exercises/ex_5_1_temperature.py`
- **Description**: See how temperature affects sampling
- **Dependencies**: MLX
- **Test**: Verify temperature changes randomness

### Exercise 5.2: Basic generation ⬜
- **File**: `exercises/ex_5_2_basic_generation.py`
- **Description**: Implement simple text generation
- **Dependencies**: MLX, model, tokenizer
- **Test**: Verify generation works (will be random for untrained)

### Exercise 5.3: Sampling comparison ⬜
- **File**: `exercises/ex_5_3_sampling_comparison.sh`
- **Description**: Compare sampling strategies
- **Dependencies**: Trained checkpoint, chat_cli_mlx.py
- **Test**: Verify different strategies produce different outputs

### Exercise 5.4: KV cache speed ⬜
- **File**: `exercises/ex_5_4_kv_cache_speed.py`
- **Description**: Measure KV cache performance
- **Dependencies**: MLX, model, KV cache
- **Test**: Verify significant speedup

---

## Phase 6: Module 6 - Training Pipeline

### Exercise 6.1: Analyze training ⬜
- **File**: `exercises/ex_6_1_analyze_training.sh`
- **Description**: Look at training logs
- **Dependencies**: Training logs or run short training
- **Test**: Verify log analysis works

### Exercise 6.2: Masked loss ⬜
- **File**: `exercises/ex_6_2_masked_loss.py`
- **Description**: Understand SFT masking
- **Dependencies**: Tokenizer
- **Test**: Verify mask shows user=0, assistant=1

### Exercise 6.3: Hyperparameters ⬜
- **File**: `exercises/ex_6_3_hyperparameters.py`
- **Description**: Document hyperparameter choices
- **Dependencies**: None (prints guide)
- **Test**: Read and verify guide is helpful

### Exercise 6.4: Checkpoint analysis ⬜
- **File**: `exercises/ex_6_4_checkpoint_analysis.py`
- **Description**: Inspect checkpoint metadata
- **Dependencies**: Checkpoints
- **Test**: Verify metadata extraction works

---

## Phase 7: Module 7 - Putting It Together

### Exercise 7.1: Train from scratch ⬜
- **File**: `exercises/ex_7_1_train_from_scratch.sh`
- **Description**: Train d2 model from scratch
- **Dependencies**: train_mlx.py
- **Test**: Verify training completes

### Exercise 7.2: Model sizes ⬜
- **File**: `exercises/ex_7_2_model_sizes.py`
- **Description**: Compare different model sizes
- **Dependencies**: MLX, model configs
- **Test**: Verify parameter counts and speed measurements

### Exercise 7.3: Evaluation ⬜
- **File**: `exercises/ex_7_3_evaluation.py`
- **Description**: Evaluate model on simple tasks
- **Dependencies**: Checkpoint, tokenizer
- **Test**: Verify evaluation runs

### Exercise 7.4: Simple chatbot ⬜
- **File**: `exercises/ex_7_4_simple_chatbot.py`
- **Description**: Build minimal chatbot
- **Dependencies**: Checkpoint, tokenizer, KV cache
- **Test**: Interactive test

---

## Implementation Order

### Stage 1: Quick Wins (Easiest)
1. Exercise 2.1: Tokenizer basics
2. Exercise 2.2: Special tokens
3. Exercise 2.3: Token efficiency
4. Exercise 3.3: Attention (conceptual)
5. Exercise 6.3: Hyperparameters (conceptual)

### Stage 2: Model Exploration
6. Exercise 3.1: Model architecture
7. Exercise 3.2: Forward pass
8. Exercise 4.1: Loss function

### Stage 3: Training & Optimization
9. Exercise 4.2: Tiny training
10. Exercise 4.3: Learning rate
11. Exercise 4.4: Checkpointing

### Stage 4: Generation & Performance
12. Exercise 5.1: Temperature
13. Exercise 5.2: Basic generation
14. Exercise 3.4: KV cache
15. Exercise 5.4: KV cache speed

### Stage 5: Advanced Topics
16. Exercise 6.2: Masked loss
17. Exercise 6.4: Checkpoint analysis
18. Exercise 7.2: Model sizes
19. Exercise 7.3: Evaluation

### Stage 6: Practical Applications
20. Exercise 7.4: Simple chatbot

### Stage 7: Shell Scripts (Need checkpoints)
21. Exercise 1.1: Run model
22. Exercise 1.2: Compare models
23. Exercise 5.3: Sampling comparison
24. Exercise 6.1: Analyze training
25. Exercise 7.1: Train from scratch

---

## Dependencies Checklist

### Required Infrastructure
- ✅ MLX installed
- ✅ Tokenizer downloaded
- ✅ NanoChat codebase working
- ⬜ Pretrained checkpoint available (d10_pretrain_causal_fix)
- ⬜ SFT checkpoint available (d10_sft)

### Python Modules Used
- `mlx.core` - MLX operations
- `mlx.nn` - Neural network layers
- `nanochat.gpt_mlx` - GPT model
- `nanochat.tokenizer` - Tokenizer
- `nanochat.kv_cache_mlx` - KV cache
- `nanochat.optimizers_mlx` - Optimizers
- `nanochat.checkpoint_mlx` - Checkpoint management

---

## Testing Strategy

### Unit Tests
Each exercise should:
1. Run without errors
2. Produce expected output format
3. Complete in reasonable time (<2 minutes for most)

### Integration Tests
1. Run all exercises in sequence
2. Verify outputs are educational
3. Check that TODOs encourage exploration

### User Testing
1. Have someone unfamiliar with LLMs try the course
2. Collect feedback on clarity
3. Improve based on feedback

---

## Next Steps

1. **Start with Stage 1** (Quick Wins) - 5 exercises
2. **Test each exercise** as we build
3. **Update this file** with ✅ as we complete
4. **Move to Stage 2** once Stage 1 is done
5. **Continue through all 7 stages**

---

## Estimated Timeline

- **Stage 1** (Quick Wins): ~1 hour
- **Stage 2** (Model Exploration): ~1 hour
- **Stage 3** (Training): ~1.5 hours
- **Stage 4** (Generation): ~1.5 hours
- **Stage 5** (Advanced): ~2 hours
- **Stage 6** (Applications): ~1 hour
- **Stage 7** (Shell scripts): ~1 hour

**Total**: ~9 hours of implementation + testing

---

## Quality Checklist

For each exercise:
- [ ] Code runs without errors
- [ ] Comments explain what's happening
- [ ] Output is clear and educational
- [ ] TODOs encourage exploration
- [ ] Referenced code files/lines are correct
- [ ] Exercise aligns with COURSE_PLAN.md description

---

## 📊 Implementation Status

**Stage 1 Complete**: 5/28 exercises (18%) ✅

**Completed Exercises**:
- ✅ ex_2_1_tokenizer_basics.py - Tokenization fundamentals
- ✅ ex_2_2_special_tokens.py - Conversation formatting
- ✅ ex_2_3_token_efficiency.py - Multi-language tokenization
- ✅ ex_3_3_attention.py - Conceptual understanding
- ✅ ex_6_3_hyperparameters.py - Hyperparameter guide

**Deferred**: Remaining 23 exercises (Stages 2-7)

**Reason for Deferral**: Course plan and first batch of exercises complete. Remaining exercises can be implemented when needed.

---

## Future Enhancements

When continuing implementation:
1. **Exercises**: Complete Stages 2-7 (23 remaining exercises)
2. **Diagrams**: Create 4 visual diagrams (see `diagrams/TASKS.md`)
3. **Notebooks**: Convert exercises to Jupyter notebooks (optional)
4. **Assessments**: Add quiz questions and automated grading
5. **Answer Keys**: Create solutions for exercise TODOs
6. **Testing Suite**: Build automated tests for all exercises
7. **Difficulty Ratings**: Add learning difficulty indicators
8. **Progress Tracker**: Create checklist/badge system

**Note**: Diagrams and videos removed from immediate plan per user request.

---

**Course plan is ready for use! 🚀**

**To continue**: Pick any stage from COURSE_PLAN_TASKS.md and implement remaining exercises.
