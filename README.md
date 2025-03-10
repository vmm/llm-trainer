# LLM Trainer

A modular framework for fine-tuning Large Language Models (LLMs) across various domains.

## Features

- Modular architecture for dataset processing, training, and evaluation
- Support for multiple models, including Llama 3
- Parameter-efficient fine-tuning with QLoRA
- Configurable hyperparameters
- Support for multiple compute environments (Google Colab, local, cloud)

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
python -m src.evaluators.reasoning_evaluator
```

## Project Structure

```
llm-trainer/
├── configs/                  # Configuration files for different experiments
├── data/                     # Data storage (processed datasets)
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── scripts/                  # Utility scripts (data downloading, conversion, etc.)
├── src/                      # Source code
│   ├── data_processors/      # Dataset processing modules
│   ├── trainers/             # Training implementation modules
│   ├── evaluators/           # Evaluation modules
│   └── utils/                # Utility functions and classes
└── requirements.txt          # Project dependencies
```

## License

MIT