#!/usr/bin/env python
"""
Main entry point for the LLM Trainer.

This script provides a command-line interface to the various components
of the LLM Trainer framework.
"""

import os
import sys
import argparse
from typing import List, Optional, Dict, Any

from src.utils.config import load_config
from src.data_processors import ReasoningDataProcessor
from src.trainers import QLoraTrainer
from src.evaluators import ReasoningEvaluator


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="LLM Trainer: A framework for fine-tuning large language models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process dataset command
    process_parser = subparsers.add_parser(
        "process", help="Process a dataset for fine-tuning"
    )
    process_parser.add_argument(
        "--config", 
        type=str, 
        default="configs/llama3_reasoning.yaml",
        help="Path to the configuration file"
    )
    process_parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where to save the processed dataset"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Fine-tune a model"
    )
    train_parser.add_argument(
        "config_path", 
        type=str, 
        help="Path to the configuration file"
    )
    train_parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the preprocessed dataset"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a fine-tuned model"
    )
    eval_parser.add_argument(
        "--config", 
        type=str, 
        default="configs/llama3_reasoning.yaml",
        help="Path to the configuration file"
    )
    eval_parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model or adapter to evaluate"
    )
    eval_parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the evaluation dataset"
    )
    eval_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )
    
    # Use command (wrapper for the script)
    use_parser = subparsers.add_parser(
        "use", help="Use a fine-tuned model"
    )
    use_parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter"
    )
    use_parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model ID/path (if not specified, will be inferred from adapter config)"
    )
    use_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 8-bit quantization (requires more GPU RAM but faster)"
    )
    use_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run model on CPU (slower but doesn't require GPU)"
    )
    use_parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    use_parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate"
    )
    use_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    return parser.parse_args(args)


def run_process_command(args: argparse.Namespace) -> None:
    """Run the process dataset command."""
    
    config = load_config(args.config)
    
    processor = ReasoningDataProcessor(config)
    processed_dataset = processor.prepare_dataset()
    
    output_path = args.output_path
    if output_path:
        processor.save_processed_dataset(output_path)
        print(f"Processed dataset saved to {output_path}")
    

def run_train_command(args: argparse.Namespace) -> None:
    """Run the train command."""
    
    config = load_config(args.config_path)
    
    trainer = QLoraTrainer(config)
    trainer.load_model()
    
    # Load the dataset
    dataset_path = args.dataset_path
    trainer.load_dataset(dataset_path)
    
    # Run training
    train_result = trainer.train()
    
    # Print training results
    print(f"Training completed. Results: {train_result}")
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


def run_evaluate_command(args: argparse.Namespace) -> None:
    """Run the evaluate command."""
    
    config = load_config(args.config)
    
    # Update config with command line arguments
    if args.output_dir:
        if "evaluation" not in config:
            config["evaluation"] = {}
        config["evaluation"]["output_dir"] = args.output_dir
    
    evaluator = ReasoningEvaluator(config)
    results = evaluator.evaluate(args.model_path, args.dataset_path)
    
    print("Evaluation complete.")


def run_use_command(args: argparse.Namespace) -> None:
    """Run the use command."""
    
    # This just calls the script
    script_path = os.path.join("scripts", "use_finetuned_model.py")
    
    # Convert args back to command line arguments
    cmd_args = [
        sys.executable,
        script_path,
        f"--adapter_path={args.adapter_path}"
    ]
    
    # Add optional arguments
    if args.base_model:
        cmd_args.append(f"--base_model={args.base_model}")
    if args.quantize:
        cmd_args.append("--quantize")
    if args.cpu:
        cmd_args.append("--cpu")
    if args.interactive:
        cmd_args.append("--interactive")
    cmd_args.append(f"--max_new_tokens={args.max_new_tokens}")
    cmd_args.append(f"--temperature={args.temperature}")
    
    # Execute the script
    os.execv(sys.executable, cmd_args)


def main() -> None:
    """Main entry point."""
    
    args = parse_args()
    
    # Dispatch to the appropriate command
    if args.command == "process":
        run_process_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "evaluate":
        run_evaluate_command(args)
    elif args.command == "use":
        run_use_command(args)
    else:
        print("Please specify a command. Run 'python main.py -h' for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()