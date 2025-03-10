"""Base trainer module for LLM fine-tuning."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import datasets
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling
)

from src.utils.config import get_config_value


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    
    A trainer is responsible for loading a model and dataset, setting up the
    training configuration, and running the training process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.model_config = get_config_value(config, "model", {})
        self.training_config = get_config_value(config, "training", {})
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize objects that will be populated by child classes
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.data_collator = None
        self.trainer = None
        
    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model to be fine-tuned.
        This method should be implemented by subclasses.
        
        Returns:
            Loaded model.
        """
        pass
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        
        Returns:
            Loaded tokenizer.
        """
        model_id = get_config_value(self.model_config, "base_model_id", None)
        if model_id is None:
            raise ValueError("Model ID must be specified in the configuration.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=get_config_value(self.model_config, "trust_remote_code", True),
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> DatasetDict:
        """
        Load the preprocessed dataset for training.
        
        Args:
            dataset_path: Path to the preprocessed dataset.
            
        Returns:
            Loaded dataset.
        """
        if dataset_path is None:
            dataset_name = get_config_value(self.config, "dataset.dataset_name", "").split("/")[-1]
            dataset_path = os.path.join("data", f"{dataset_name}_processed")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Please process the dataset first using a DataProcessor."
            )
        
        self.dataset = datasets.load_from_disk(dataset_path)
        return self.dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """
        Set up the training arguments for the Trainer.
        
        Returns:
            TrainingArguments object.
        """
        # Extract training configuration
        output_dir = get_config_value(self.training_config, "output_dir", "./output")
        num_train_epochs = get_config_value(self.training_config, "num_train_epochs", 3)
        per_device_train_batch_size = get_config_value(self.training_config, "per_device_train_batch_size", 4)
        per_device_eval_batch_size = get_config_value(self.training_config, "per_device_eval_batch_size", 4)
        gradient_accumulation_steps = get_config_value(self.training_config, "gradient_accumulation_steps", 8)
        
        learning_rate = get_config_value(self.training_config, "learning_rate", 2e-4)
        weight_decay = get_config_value(self.training_config, "weight_decay", 0.01)
        lr_scheduler_type = get_config_value(self.training_config, "lr_scheduler_type", "cosine")
        warmup_ratio = get_config_value(self.training_config, "warmup_ratio", 0.03)
        optim = get_config_value(self.training_config, "optim", "paged_adamw_32bit")
        
        fp16 = get_config_value(self.training_config, "fp16", True)
        bf16 = get_config_value(self.training_config, "bf16", False)
        
        evaluation_strategy = get_config_value(self.training_config, "evaluation_strategy", "steps")
        eval_steps = get_config_value(self.training_config, "eval_steps", 200)
        save_strategy = get_config_value(self.training_config, "save_strategy", "steps")
        save_steps = get_config_value(self.training_config, "save_steps", 200)
        save_total_limit = get_config_value(self.training_config, "save_total_limit", 3)
        
        logging_steps = get_config_value(self.training_config, "logging_steps", 50)
        log_level = get_config_value(self.training_config, "log_level", "info")
        
        push_to_hub = get_config_value(self.training_config, "push_to_hub", False)
        hub_model_id = get_config_value(self.training_config, "hub_model_id", None)
        hub_token = get_config_value(self.training_config, "hub_token", None)
        
        report_to = get_config_value(self.training_config, "report_to", "tensorboard")
        seed = get_config_value(self.training_config, "seed", 42)
        dataloader_num_workers = get_config_value(self.training_config, "dataloader_num_workers", 4)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optim=optim,
            fp16=fp16,
            bf16=bf16,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            log_level=log_level,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_token=hub_token,
            report_to=report_to,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
        )
        
        return training_args
    
    def setup_data_collator(self) -> DataCollatorForLanguageModeling:
        """
        Set up the data collator for the Trainer.
        
        Returns:
            Data collator.
        """
        if self.tokenizer is None:
            self.load_tokenizer()
            
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )
        
        return self.data_collator
    
    @abstractmethod
    def setup_trainer(self) -> Trainer:
        """
        Set up the Trainer.
        This method should be implemented by subclasses.
        
        Returns:
            Trainer object.
        """
        pass
    
    def train(self) -> Any:
        """
        Run the training process.
        
        Returns:
            Training results.
        """
        if self.trainer is None:
            self.setup_trainer()
            
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model, tokenizer, and config
        self.trainer.save_model()
        self.trainer.save_state()
        
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Returns:
            Evaluation results.
        """
        if self.trainer is None:
            self.setup_trainer()
            
        eval_results = self.trainer.evaluate()
        
        return eval_results