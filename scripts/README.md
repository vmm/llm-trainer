# Utility Scripts

This directory contains utility scripts for the LLM Trainer project.

## Available Scripts

### `use_finetuned_model.py`

Script to use a fine-tuned model with LoRA adapters locally.

#### Usage

```bash
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter

# Run in interactive mode
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter --interactive

# Use a specific base model (if different from the one in adapter config)
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter --base_model meta-llama/Meta-Llama-3-8B

# Use quantization for faster inference on GPU
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter --quantize

# Run on CPU (slower but doesn't require GPU)
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter --cpu

# Additional generation options
python scripts/use_finetuned_model.py --adapter_path /path/to/adapter --max_new_tokens 256 --temperature 0.8
```

#### Arguments

- `--adapter_path`: Path to the trained LoRA adapter (required)
- `--base_model`: Base model ID/path (if not specified, will be inferred from adapter config)
- `--quantize`: Load model in 8-bit quantization (requires more GPU RAM but faster)
- `--cpu`: Run model on CPU (slower but doesn't require GPU)
- `--interactive`: Run in interactive mode
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 128)
- `--temperature`: Sampling temperature (default: 0.7)

## Adding New Scripts

When adding new utility scripts to this directory:

1. Make the script executable with `chmod +x script_name.py`
2. Add documentation in this README
3. Follow the existing pattern of using argparse for command-line arguments