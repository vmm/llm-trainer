# LLM Trainer Framework Summary

This document provides a high-level overview of the LLM Trainer framework, a modular system for fine-tuning Large Language Models (LLMs) with an initial focus on enhancing reasoning capabilities.

## Architecture Overview

The framework follows a modular architecture with these key components:

### 1. Data Processing

- `BaseDataProcessor`: Abstract base class for all data processors
- `ReasoningDataProcessor`: Specialized processor for reasoning datasets
- Supports loading, preprocessing, tokenizing, and saving datasets

### 2. Training

- `BaseTrainer`: Abstract base class for all trainers
- `QLoraTrainer`: Implementation of QLoRA (Quantized Low-Rank Adaptation) training
- Supports 4-bit quantization, gradient checkpointing, and efficient fine-tuning

### 3. Evaluation

- `BaseEvaluator`: Abstract base class for all evaluators
- `ReasoningEvaluator`: Specialized evaluator for reasoning tasks
- Supports loading models, running inference, and calculating metrics

### 4. Deployment & Usage

- Interactive CLI for model interaction
- Gradio web interface for user-friendly interaction
- Model export tools for production deployment (ONNX, merged models)
- Adapter management utilities

### 5. Utilities

- Configuration management with YAML files
- Google Drive integration for Colab persistence
- Standardized CLI for all components
- Helper scripts for using fine-tuned models

## Current Implementation

The initial implementation focuses on:

- **Model**: Llama 3 8B
- **Task**: Enhanced reasoning
- **Dataset**: facebook/natural_reasoning
- **Technique**: QLoRA with 4-bit quantization
- **Sequence length**: 2048 tokens

## Key Features

- Modular and extensible architecture
- Parameter-efficient fine-tuning with QLoRA
- Google Colab compatibility with Drive integration
- Multiple deployment options for fine-tuned models
- Interactive web interface with Gradio
- ONNX export for production deployment
- Adapter merging with base model
- Robust error handling and fallbacks

## Usage Flow

### Training Pipeline
1. **Process Dataset**: `python -m src.data_processors.reasoning_processor`
2. **Fine-tune Model**: `python -m src.trainers.qlora_trainer configs/llama3_reasoning.yaml`
3. **Evaluate Model**: `python -m src.evaluators.reasoning_evaluator --model_path ./output/llama3_reasoning`

### Using Fine-tuned Models
- **Interactive CLI**: `python -m scripts.use_finetuned_model --adapter_path ./output/llama3_reasoning --interactive`
- **Web Interface**: `python -m scripts.use_finetuned_model --adapter_path ./output/llama3_reasoning --gradio`
- **Merge Adapter**: `python -m scripts.use_finetuned_model --adapter_path ./output/llama3_reasoning --merge_adapter --output_dir merged_model`
- **Export to ONNX**: `python -m scripts.export_model_onnx --model_path merged_model --output_dir onnx_model --quantize`

## Google Colab Integration

Two Jupyter notebooks are included for running the training process on Google Colab with GPU acceleration:
- `notebooks/llama3_reasoning_finetuning.ipynb` - Basic training notebook
- `notebooks/llama3_reasoning_finetuning_drive.ipynb` - Enhanced notebook with Google Drive integration for persistently storing models, datasets, and evaluations

## Extension Points

The framework is designed for easy extension:

1. **New Data Processors**: Create new subclasses of `BaseDataProcessor` for different datasets
2. **New Training Methods**: Implement alternatives to QLoRA by subclassing `BaseTrainer`
3. **New Evaluation Methods**: Add specialized evaluators by subclassing `BaseEvaluator`
4. **New Models**: Update configuration files to support different base models
5. **New Interfaces**: Add visualization or interaction interfaces that leverage the trained models

## Recent Enhancements

- Added Gradio web interface for easy interaction with models
- Implemented LoRA adapter merging for deployment efficiency
- Added ONNX export for optimized inference in production
- Improved Google Drive integration for Colab persistence
- Fixed issues with Flash Attention compatibility
- Added fallback mechanisms for dataset splits
- Enhanced error handling throughout the pipeline

## Roadmap

For a detailed development roadmap including planned features and enhancements, please see the [ROADMAP.md](ROADMAP.md) file.