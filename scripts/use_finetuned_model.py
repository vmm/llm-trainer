#!/usr/bin/env python
"""
Script to use a fine-tuned model with LoRA adapters locally.

This script loads a fine-tuned model and runs inference on test questions.
"""

import os
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Use a fine-tuned LLM locally")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model ID/path (if not specified, will be inferred from adapter config)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 8-bit quantization (requires more GPU RAM but faster)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run model on CPU (slower but doesn't require GPU)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(
    adapter_path: str,
    base_model: Optional[str] = None,
    quantize: bool = False,
    use_cpu: bool = False
) -> tuple:
    """Load the model and tokenizer."""
    
    # Load adapter config
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Use provided base model or get from config
    base_model_id = base_model or config.base_model_name_or_path
    
    print(f"Loading base model: {base_model_id}")
    
    # Load model with appropriate settings
    device_map = "cpu" if use_cpu else "auto"
    
    if quantize and not use_cpu:
        # Load in 8-bit precision for efficiency
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            load_in_8bit=True,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        # Load in full precision or on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # Load LoRA adapter
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_pipeline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    use_cpu: bool = False,
    **generation_kwargs: Dict[str, Any]
) -> pipeline:
    """Create a text generation pipeline."""
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=("cpu" if use_cpu else 0),
        **generation_kwargs
    )


def generate_answer(
    pipe: pipeline,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7
) -> str:
    """Generate an answer for a given question."""
    
    prompt = f"Question: {question}\n\nAnswer: "
    
    result = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    return result[0]["generated_text"].strip()


def interactive_mode(
    pipe: pipeline,
    max_new_tokens: int = 128,
    temperature: float = 0.7
) -> None:
    """Run the model in interactive mode."""
    
    print("\n" + "=" * 50)
    print("Interactive Question Answering")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50 + "\n")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
            
        answer = generate_answer(
            pipe, 
            question, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
        
        print(f"\nAnswer: {answer}")
        print("-" * 80)


def main():
    """Main function."""
    
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.adapter_path,
        args.base_model,
        args.quantize,
        args.cpu
    )
    
    # Create pipeline
    pipe = create_pipeline(
        model,
        tokenizer,
        args.cpu
    )
    
    if args.interactive:
        # Run in interactive mode
        interactive_mode(
            pipe,
            args.max_new_tokens,
            args.temperature
        )
    else:
        # Run on default test questions
        test_questions = [
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "If no mammals can fly, and all bats can fly, what can we conclude about bats?",
            "If all A are B, and all B are C, what can we conclude about the relationship between A and C?"
        ]
        
        for question in test_questions:
            answer = generate_answer(
                pipe, 
                question, 
                args.max_new_tokens, 
                args.temperature
            )
            
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print("-" * 80)


if __name__ == "__main__":
    main()