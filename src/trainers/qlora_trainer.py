"""QLoRA trainer for fine-tuning LLMs."""

import os
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from src.trainers.base_trainer import BaseTrainer
from src.utils.config import get_config_value, load_config


class QLoraTrainer(BaseTrainer):
    """
    Trainer for QLoRA fine-tuning of LLMs.
    
    QLoRA (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning
    technique that uses 4-bit quantization and low-rank adapters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the QLoRA trainer.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.lora_config = get_config_value(config, "lora", {})
        
    def load_model(self) -> Union[AutoModelForCausalLM, PeftModel]:
        """
        Load the model with quantization and prepare it for QLoRA fine-tuning.
        
        Returns:
            Quantized model with LoRA adapters.
        """
        model_id = get_config_value(self.model_config, "base_model_id", None)
        if model_id is None:
            raise ValueError("Model ID must be specified in the configuration.")
        
        # Load existing adapter if specified
        adapter_path = get_config_value(self.model_config, "adapter_name_or_path", None)
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading existing adapter from {adapter_path}")
            
            # Load the base model with quantization
            self._load_base_model_with_quantization(model_id)
            
            # Load the adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=True,
            )
            return self.model
        
        # Load the base model with quantization
        self._load_base_model_with_quantization(model_id)
        
        # Prepare model for kbit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )
        
        # Configure LoRA adapter
        self._setup_lora_adapter()
        
        return self.model
    
    def _load_base_model_with_quantization(self, model_id: str) -> AutoModelForCausalLM:
        """
        Load the base model with 4-bit quantization.
        
        Args:
            model_id: ID of the model to load.
            
        Returns:
            Quantized model.
        """
        load_in_4bit = get_config_value(self.model_config, "load_in_4bit", True)
        use_flash_attention = get_config_value(self.model_config, "use_flash_attention", True)
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load model with quantization config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=get_config_value(self.model_config, "trust_remote_code", True),
            device_map="auto",
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        )
        
        return self.model
    
    def _setup_lora_adapter(self) -> PeftModel:
        """
        Set up the LoRA adapter for fine-tuning.
        
        Returns:
            Model with LoRA adapter.
        """
        # Get LoRA configuration
        lora_r = get_config_value(self.lora_config, "r", 16)
        lora_alpha = get_config_value(self.lora_config, "lora_alpha", 32)
        lora_dropout = get_config_value(self.lora_config, "lora_dropout", 0.05)
        bias = get_config_value(self.lora_config, "bias", "none")
        task_type = get_config_value(self.lora_config, "task_type", "CAUSAL_LM")
        target_modules = get_config_value(
            self.lora_config, 
            "target_modules", 
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=getattr(TaskType, task_type),
            target_modules=target_modules,
        )
        
        # Apply LoRA adapter to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters info
        self.model.print_trainable_parameters()
        
        return self.model
    
    def setup_trainer(self) -> Trainer:
        """
        Set up the Trainer for QLoRA fine-tuning.
        
        Returns:
            Configured Trainer.
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
            
        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            self.load_tokenizer()
            
        # Load dataset if not already loaded
        if self.dataset is None:
            self.load_dataset()
            
        # Set up data collator if not already set up
        if self.data_collator is None:
            self.setup_data_collator()
            
        # Set up training arguments
        training_args = self.setup_training_arguments()
        
        # Create the trainer
        train_split = get_config_value(self.config, "dataset.train_split", "train")
        eval_split = get_config_value(self.config, "dataset.validation_split", "validation")
        
        # Make sure splits exist, fallback to available splits if needed
        available_splits = list(self.dataset.keys())
        if train_split not in available_splits:
            raise ValueError(f"Train split '{train_split}' not found in dataset. Available splits: {available_splits}")
        
        # For eval_split, try to find a suitable alternative if the specified one doesn't exist
        if eval_split not in available_splits:
            # Try common validation split names
            for alternative in ["validation", "val", "valid", "dev", "test"]:
                if alternative in available_splits:
                    print(f"Validation split '{eval_split}' not found, using '{alternative}' instead.")
                    eval_split = alternative
                    break
        
        # Create trainer with appropriate splits
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": self.dataset[train_split],
            "tokenizer": self.tokenizer,
            "data_collator": self.data_collator,
        }
        
        # Add eval dataset if available
        if eval_split in available_splits:
            trainer_kwargs["eval_dataset"] = self.dataset[eval_split]
        else:
            print(f"Warning: No validation split found in dataset. Training without evaluation.")
            # Disable evaluation in training args
            training_args.evaluation_strategy = "no"
        
        self.trainer = Trainer(**trainer_kwargs)
        
        return self.trainer


if __name__ == "__main__":
    """Run the trainer as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune an LLM using QLoRA")
    parser.add_argument(
        "config_path", 
        type=str, 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the preprocessed dataset"
    )
    
    args = parser.parse_args()
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