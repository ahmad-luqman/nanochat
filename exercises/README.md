# NanoChat Learning Exercises

This directory contains hands-on exercises to help you understand how LLMs work.

## Structure

Exercises are organized by module (matching `COURSE_PLAN.md`):

```
exercises/
├── Module 1: Big Picture
│   ├── ex_1_1_*.py
│   └── ex_1_2_*.py
├── Module 2: Tokenization
│   ├── ex_2_1_tokenizer_basics.py
│   ├── ex_2_2_special_tokens.py
│   └── ex_2_3_token_efficiency.py
├── Module 3: Architecture
│   ├── ex_3_1_model_architecture.py
│   ├── ex_3_2_forward_pass.py
│   ├── ex_3_3_attention.py
│   └── ex_3_4_kv_cache.py
├── Module 4: Training
│   ├── ex_4_1_loss_function.py
│   ├── ex_4_2_tiny_training.py
│   ├── ex_4_3_learning_rate.py
│   └── ex_4_4_checkpointing.py
├── Module 5: Generation
│   ├── ex_5_1_temperature.py
│   ├── ex_5_2_basic_generation.py
│   ├── ex_5_3_sampling_comparison.sh
│   └── ex_5_4_kv_cache_speed.py
├── Module 6: Training Pipeline
│   ├── ex_6_1_analyze_training.sh
│   ├── ex_6_2_masked_loss.py
│   ├── ex_6_3_hyperparameters.py
│   └── ex_6_4_checkpoint_analysis.py
└── Module 7: Putting It Together
    ├── ex_7_1_train_from_scratch.sh
    ├── ex_7_2_model_sizes.py
    ├── ex_7_3_evaluation.py
    └── ex_7_4_simple_chatbot.py
```

## How to Use

1. **Read the Course Plan**: Start with `COURSE_PLAN.md` to understand the learning path

2. **Follow the Modules**: Complete exercises in order (Module 1 → Module 7)

3. **Run the Exercises**: Each exercise is a standalone script:
   ```bash
   # Python exercises
   python exercises/ex_2_1_tokenizer_basics.py

   # Shell exercises
   bash exercises/ex_5_3_sampling_comparison.sh
   ```

4. **Modify and Experiment**: Each exercise has TODOs - try them!

5. **Ask Questions**: If stuck, refer back to the course plan or code references

## Prerequisites

- Python 3.10+
- MLX installed (`pip install mlx`)
- NanoChat dependencies (`uv sync` or `pip install -r requirements.txt`)
- Tokenizer downloaded (runs automatically when you use the tokenizer)

## Tips

- **Start Simple**: Don't skip the early exercises - they build foundational understanding
- **Read the Code**: Look at the referenced code files in `nanochat/` and `scripts/`
- **Experiment**: Change parameters and see what happens
- **Take Notes**: Document your observations
- **Debug**: Use `print()` statements to understand what's happening

## Getting Help

If you're stuck:

1. Re-read the ELI5 explanation for that module
2. Look at the code references (file:line format)
3. Try simpler versions of the exercise
4. Check if there are syntax errors or missing imports

## Output

Exercise outputs (checkpoints, logs, etc.) will be saved to:
- `exercises/my_first_model/` - Your trained model
- `exercises/test_sft/` - Test fine-tuning runs
- Exercise-specific output files

These are gitignored - they're for your learning only!

## Next Steps

After completing all exercises:
1. Try the Final Project Ideas in `COURSE_PLAN.md`
2. Build your own application
3. Share what you learned!

Happy learning! 🚀
