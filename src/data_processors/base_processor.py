"""Base data processor for LLM training datasets."""

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import datasets
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer

from src.utils.config import get_config_value
from src.utils.auth import setup_huggingface_auth, get_auth_token, requires_authentication


class BaseDataProcessor(ABC):
    """
    Base class for all data processors.
    
    A data processor is responsible for loading a dataset, preprocessing it,
    and preparing it for training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.dataset_config = get_config_value(config, "dataset", {})
        self.model_config = get_config_value(config, "model", {})
        
        # Setup HuggingFace authentication
        hf_token = get_config_value(self.model_config, "hf_token", None)
        use_auth_token = get_config_value(self.model_config, "use_auth_token", True)
        self.auth_successful = setup_huggingface_auth(hf_token, use_auth_token)
        
        # Initialize tokenizer
        tokenizer_id = get_config_value(
            self.model_config, 
            "base_model_id", 
            "meta-llama/Meta-Llama-3-8B"
        )
        
        # Check if model requires authentication
        if requires_authentication(tokenizer_id) and not self.auth_successful:
            print(f"Warning: Model {tokenizer_id} may require authentication. "
                  f"Consider setting HF_TOKEN environment variable or adding hf_token to config.")
        
        # Load tokenizer with authentication
        tokenizer_kwargs = {
            "trust_remote_code": get_config_value(self.model_config, "trust_remote_code", True),
        }
        
        # Add token if authentication was successful
        if self.auth_successful and hf_token:
            tokenizer_kwargs["token"] = hf_token
        elif self.auth_successful:
            auth_token = get_auth_token()
            if auth_token:
                tokenizer_kwargs["token"] = auth_token
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            **tokenizer_kwargs
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set dataset attributes
        self.dataset_name = get_config_value(self.dataset_config, "dataset_name", None)
        self.train_split = get_config_value(self.dataset_config, "train_split", "train")
        self.validation_split = get_config_value(self.dataset_config, "validation_split", "validation")
        self.text_column = get_config_value(self.dataset_config, "text_column", "text")
        self.max_seq_length = get_config_value(self.dataset_config, "max_seq_length", 2048)
        
        # Dataset objects will be populated by load_dataset
        self.raw_dataset = None
        self.processed_dataset = None
    
    def load_dataset(self) -> DatasetDict:
        """
        Load the dataset from the configured source.
        
        Returns:
            Loaded dataset dictionary.
        """
        if self.dataset_name is None:
            raise ValueError("Dataset name must be specified in the configuration.")
        
        # Load dataset from Hugging Face
        self.raw_dataset = datasets.load_dataset(self.dataset_name)
        
        return self.raw_dataset
    
    @abstractmethod
    def preprocess_dataset(self) -> DatasetDict:
        """
        Preprocess the dataset for the specific task.
        This method should be implemented by subclasses.
        
        Returns:
            Processed dataset.
        """
        pass
    
    def tokenize_dataset(
        self, 
        dataset: Dataset, 
        text_column: Optional[str] = None,
    ) -> Dataset:
        """
        Tokenize the dataset using the model's tokenizer.
        
        Args:
            dataset: Dataset to tokenize.
            text_column: Column containing the text to tokenize.
            
        Returns:
            Tokenized dataset.
        """
        text_column = text_column or self.text_column
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
        
        # Apply tokenization to the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != text_column],
        )
        
        return tokenized_dataset
    
    def save_processed_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Save the processed dataset to disk.
        
        Args:
            output_path: Path where to save the dataset.
            
        Returns:
            Path where the dataset was saved.
        """
        if self.processed_dataset is None:
            raise ValueError("Dataset has not been processed yet. Call preprocess_dataset first.")
        
        if output_path is None:
            dataset_name = self.dataset_name.split("/")[-1]
            output_path = os.path.join("data", f"{dataset_name}_processed")
        
        os.makedirs(output_path, exist_ok=True)
        self.processed_dataset.save_to_disk(output_path)
        
        return output_path
    
    def load_processed_dataset(self, input_path: str) -> DatasetDict:
        """
        Load a previously processed dataset from disk.
        
        Args:
            input_path: Path to the processed dataset.
            
        Returns:
            Loaded dataset.
        """
        self.processed_dataset = datasets.load_from_disk(input_path)
        return self.processed_dataset
    
    def prepare_dataset(self) -> DatasetDict:
        """
        End-to-end dataset preparation:
        1. Load the dataset
        2. Preprocess it
        3. Save it to disk
        
        Returns:
            Processed dataset.
        """
        self.load_dataset()
        self.processed_dataset = self.preprocess_dataset()
        self.save_processed_dataset()
        
        return self.processed_dataset