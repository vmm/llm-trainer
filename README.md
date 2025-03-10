# LLM Trainer

A modular framework for fine-tuning Large Language Models (LLMs) across various domains.

## Features

- Modular architecture for dataset processing, training, and evaluation
- Support for multiple models, including Llama 3
- Parameter-efficient fine-tuning with QLoRA
- Configurable hyperparameters
- Support for multiple compute environments (Google Colab, local, cloud)
- Interactive web interface with Gradio
- ONNX export for production deployment
- Adapter merging for deployment efficiency
- Google Drive integration for Colab persistence

## Initial Implementation

- Model: Llama 3 8B
- Task: Enhanced reasoning
- Dataset: facebook/natural_reasoning
- Technique: QLoRA with 4-bit quantization
- Sequence length: 2048 tokens

## Setup

```bash
# Clone the repository
git clone https://github.com/vmm/llm-trainer.git
cd llm-trainer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python -m src.data_processors.reasoning_processor
```

### Training

```bash
python -m src.trainers.qlora_trainer configs/llama3_reasoning.yaml
```

### Evaluation

```bash
python -m src.evaluators.reasoning_evaluator --config configs/llama3_reasoning.yaml --model_path output/llama3_reasoning
```

### Using the Fine-tuned Model

You can use the fine-tuned model in several ways:

#### Command Line Interface

```bash
python -m scripts.use_finetuned_model --adapter_path output/llama3_reasoning/adapter_model --quantize
```

#### Interactive Mode

```bash
python -m scripts.use_finetuned_model --adapter_path output/llama3_reasoning/adapter_model --interactive
```

#### Web Interface with Gradio

```bash
python -m scripts.use_finetuned_model --adapter_path output/llama3_reasoning/adapter_model --gradio
```

### Model Export and Deployment

#### Merge LoRA Adapter with Base Model

```bash
python -m scripts.use_finetuned_model --adapter_path output/llama3_reasoning/adapter_model --merge_adapter --output_dir merged_model
```

#### Export to ONNX for Production

```bash
python -m scripts.export_model_onnx --model_path merged_model --output_dir onnx_model --quantize
```

## Colab Notebook with Drive Integration

The project includes a Colab notebook with Google Drive integration for persistent storage:

- `notebooks/llama3_reasoning_finetuning_drive.ipynb`

This notebook sets up everything needed to fine-tune models in Colab and automatically saves all results to your Google Drive.

## Project Structure

```
llm-trainer/
├── configs/                  # Configuration files for different experiments
├── data/                     # Data storage (processed datasets)
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── scripts/                  # Utility scripts (data downloading, conversion, etc.)
│   ├── use_finetuned_model.py   # Script for using the fine-tuned model
│   └── export_model_onnx.py    # Script for exporting to ONNX format
├── src/                      # Source code
│   ├── data_processors/      # Dataset processing modules
│   ├── trainers/             # Training implementation modules
│   ├── evaluators/           # Evaluation modules
│   └── utils/                # Utility functions and classes
└── requirements.txt          # Project dependencies
```

## License

MIT