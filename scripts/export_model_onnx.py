#!/usr/bin/env python
"""
Script to export a fine-tuned model to ONNX format for optimized inference.

This script takes a fine-tuned model (either the base model with adapter or a merged model)
and exports it to ONNX format for faster inference in production environments.
"""

import os
import argparse
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export a fine-tuned model to ONNX format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or adapter to export"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the exported ONNX model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model ID/path (if exporting an adapter)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the ONNX model to int8 (reduces size but may affect accuracy)"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length for the ONNX model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for export (cuda, cpu)"
    )

    return parser.parse_args()


def load_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    device: str = "cuda"
) -> tuple:
    """
    Load the model for ONNX export.
    
    Args:
        model_path: Path to the model or adapter
        base_model_path: Path to the base model (if model_path is an adapter)
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if model_path is an adapter
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_model.bin")) or \
                os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    tokenizer = None
    model = None
    
    if is_adapter and base_model_path:
        print(f"Loading adapter from {model_path} with base model {base_model_path}")
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load the adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Optional: merge for export
        model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
    elif is_adapter and not base_model_path:
        # Try to infer base model from adapter config
        print(f"Loading adapter config to infer base model")
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        
        print(f"Using base model: {base_model_path}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Merge for export
        model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
    else:
        # Regular model (not an adapter)
        print(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def export_model_to_onnx(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    sequence_length: int = 2048,
    quantize: bool = False
) -> str:
    """
    Export the model to ONNX format.
    
    Args:
        model: Model to export
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the exported model
        sequence_length: Maximum sequence length for the ONNX model
        quantize: Whether to quantize the model to int8
        
    Returns:
        Path to the exported ONNX model
    """
    try:
        from optimum.exporters import OnnxConfig
        from optimum.exporters.onnx import export_onnx, main_export
    except ImportError:
        print("Error: optimum library not installed. Install with 'pip install optimum'.")
        print("For quantization, also install 'pip install optimum[onnxruntime]'")
        return None
    
    print(f"Exporting model to ONNX format with sequence length {sequence_length}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up ONNX export configuration
    onnx_config = OnnxConfig(
        task="causal-lm",
        model_type=model.config.model_type,
        token_has_type_id=False,
        use_cache=True,
        use_past=True,
        input_shapes={"sequence_length": sequence_length}
    )
    
    # Export the model to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    # Export model
    model_path = export_onnx(
        model=model,
        tokenizer=tokenizer,
        onnx_config=onnx_config,
        output=output_dir,
        device=model.device
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Quantize if requested
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            print("Quantizing ONNX model to int8")
            onnx_quantized_path = os.path.join(output_dir, "model_quantized.onnx")
            
            quantize_dynamic(
                model_input=os.path.join(output_dir, "decoder_model.onnx"),
                model_output=onnx_quantized_path,
                weight_type=QuantType.QInt8
            )
            
            print(f"Quantized model saved to {onnx_quantized_path}")
        except ImportError:
            print("Error: onnxruntime-quantization not installed.")
            print("Install with 'pip install onnxruntime-tools'")
    
    return output_dir


def main():
    """Main function."""
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        args.base_model,
        args.device
    )
    
    # Export model
    output_path = export_model_to_onnx(
        model,
        tokenizer,
        args.output_dir,
        args.sequence_length,
        args.quantize
    )
    
    if output_path:
        print(f"Model successfully exported to ONNX at {output_path}")
    else:
        print("Error exporting model to ONNX")


if __name__ == "__main__":
    main()