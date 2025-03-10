#!/usr/bin/env python
"""
Script to use a fine-tuned model with LoRA adapters locally.

This script loads a fine-tuned model and runs inference on test questions.
It also provides a web-based interface using Gradio for easy interaction.
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
    
    # Interface modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    mode_group.add_argument(
        "--gradio", 
        action="store_true",
        help="Launch Gradio web interface"
    )
    mode_group.add_argument(
        "--merge_adapter", 
        action="store_true",
        help="Merge LoRA adapter with base model for deployment"
    )
    
    # Generation settings
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
    
    # Server settings
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save merged model (required for --merge_adapter)"
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


def launch_gradio_interface(
    pipe: pipeline,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    port: int = 7860
) -> None:
    """Launch Gradio web interface for the model."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it with 'pip install gradio'.")
        return
    
    # Get model info from pipeline
    model_name = pipe.model.config._name_or_path.split("/")[-1]
    if hasattr(pipe.model, "peft_config"):
        model_name = f"{model_name} + LoRA adapter"

    # Define interface function
    def predict(question, temperature_slider, max_tokens_slider):
        if not question:
            return "Please enter a question."
        
        prompt = f"Question: {question}\n\nAnswer: "
        
        result = pipe(
            prompt,
            max_new_tokens=int(max_tokens_slider),
            do_sample=True,
            temperature=float(temperature_slider),
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        
        return result[0]["generated_text"].strip()
    
    # Build Gradio interface
    with gr.Blocks(title=f"Fine-tuned {model_name} for Reasoning") as demo:
        gr.Markdown(f"# Fine-tuned {model_name} for Reasoning")
        gr.Markdown("Enter a reasoning question to see how the model responds.")
        
        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your reasoning question here...",
                    lines=3
                )
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=temperature,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more precise"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=16,
                        maximum=512,
                        value=max_new_tokens,
                        step=16,
                        label="Max Tokens",
                        info="Maximum length of generated response"
                    )
                
                submit_btn = gr.Button("Get Answer", variant="primary")
                
            with gr.Column(scale=6):
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    placeholder="The model's answer will appear here..."
                )
        
        # Add example questions
        example_questions = [
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "If no mammals can fly, and all bats can fly, what can we conclude about bats?",
            "If all A are B, and all B are C, what can we conclude about the relationship between A and C?",
            "If it is raining, then the ground is wet. The ground is wet. Can we conclude that it is raining?",
            "All students in the class passed the exam. John is in the class. What can we conclude about John?"
        ]
        
        gr.Examples(
            examples=example_questions,
            inputs=question_input
        )
        
        # Set up event handlers
        submit_btn.click(
            fn=predict,
            inputs=[question_input, temperature_slider, max_tokens_slider],
            outputs=answer_output
        )
        
        question_input.submit(
            fn=predict,
            inputs=[question_input, temperature_slider, max_tokens_slider],
            outputs=answer_output
        )
    
    # Launch the interface
    demo.launch(server_port=port, share=True)


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
    
    if args.gradio:
        # Launch Gradio interface
        launch_gradio_interface(
            pipe,
            args.max_new_tokens,
            args.temperature,
            args.port
        )
    elif args.interactive:
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


def merge_adapter_with_base_model(
    adapter_path: str,
    output_dir: str,
    base_model: Optional[str] = None,
    device: str = "auto"
) -> None:
    """
    Merge LoRA adapter weights with the base model for easier deployment.
    
    Args:
        adapter_path: Path to the LoRA adapter
        output_dir: Directory to save the merged model
        base_model: Base model ID or path (inferred from adapter config if None)
        device: Device to use for merging (cpu, cuda, auto)
    """
    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: Required libraries not installed.")
        return
    
    print(f"Loading adapter config from {adapter_path}")
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Use provided base model or get from config
    base_model_id = base_model or config.base_model_name_or_path
    print(f"Using base model: {base_model_id}")
    
    # Load base model (using fp16 to reduce memory footprint)
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )
    
    # Load adapter
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapter with base model
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Successfully merged adapter with base model. Saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    
    # Handle --merge_adapter mode
    if args.merge_adapter:
        if not args.output_dir:
            print("Error: --output_dir must be specified when using --merge_adapter")
            exit(1)
        
        merge_adapter_with_base_model(
            args.adapter_path,
            args.output_dir,
            args.base_model
        )
    else:
        # Run in normal mode (command line, interactive, or gradio)
        main()