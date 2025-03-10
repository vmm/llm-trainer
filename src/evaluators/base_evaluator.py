"""Base evaluator for LLM models."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import evaluate
import datasets

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils.config import get_config_value


class BaseEvaluator(ABC):
    """
    Base class for model evaluators.
    
    An evaluator loads a model and dataset, performs inference, and calculates
    evaluation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.model_config = get_config_value(config, "model", {})
        self.evaluation_config = get_config_value(config, "evaluation", {})
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize objects that will be populated later
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.dataset = None
        self.metrics = {}
        
    def load_model(self, model_path: Optional[str] = None) -> AutoModelForCausalLM:
        """
        Load the model for evaluation.
        
        Args:
            model_path: Path to the model directory or Hugging Face model ID.
                If None, will use the model specified in the configuration.
                
        Returns:
            Loaded model.
        """
        model_id = model_path or get_config_value(
            self.model_config, "base_model_id", None
        )
        
        if model_id is None:
            raise ValueError("Model ID or path must be specified.")
        
        # Load model
        load_in_4bit = get_config_value(self.model_config, "load_in_4bit", True)
        
        if load_in_4bit:
            # Load in 4-bit precision for efficiency during evaluation
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                trust_remote_code=get_config_value(self.model_config, "trust_remote_code", True),
                device_map="auto",
            )
        else:
            # Load in full precision
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=get_config_value(self.model_config, "trust_remote_code", True),
                device_map="auto",
            )
            
        return self.model
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        
        Args:
            tokenizer_path: Path to the tokenizer or Hugging Face tokenizer ID.
                If None, will use the model specified in the configuration.
                
        Returns:
            Loaded tokenizer.
        """
        tokenizer_id = tokenizer_path or get_config_value(
            self.model_config, "base_model_id", None
        )
        
        if tokenizer_id is None:
            raise ValueError("Tokenizer ID or path must be specified.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=get_config_value(self.model_config, "trust_remote_code", True),
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def setup_pipeline(self) -> pipeline:
        """
        Set up the text generation pipeline.
        
        Returns:
            Text generation pipeline.
        """
        if self.model is None:
            self.load_model()
            
        if self.tokenizer is None:
            self.load_tokenizer()
            
        # Set up text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        return self.pipeline
    
    @abstractmethod
    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Load the evaluation dataset.
        This method should be implemented by subclasses.
        
        Returns:
            Evaluation dataset.
        """
        pass
    
    def load_metrics(self) -> Dict[str, Any]:
        """
        Load evaluation metrics.
        
        Returns:
            Dictionary of metric objects.
        """
        metric_names = get_config_value(self.evaluation_config, "metrics", ["accuracy"])
        
        for metric_name in metric_names:
            self.metrics[metric_name] = evaluate.load(metric_name)
            
        return self.metrics
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the dataset.
        This method should be implemented by subclasses.
        
        Returns:
            Dictionary of evaluation results.
        """
        pass
    
    def generate_predictions(
        self, 
        texts: List[str], 
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate predictions for a list of input texts.
        
        Args:
            texts: List of input texts.
            generation_kwargs: Additional arguments for text generation.
            
        Returns:
            List of generated texts.
        """
        if self.pipeline is None:
            self.setup_pipeline()
            
        # Get generation config
        gen_kwargs = generation_kwargs or get_config_value(
            self.evaluation_config, "generate_kwargs", {}
        )
        
        # Generate predictions
        outputs = []
        for text in texts:
            result = self.pipeline(
                text,
                return_full_text=False,
                **gen_kwargs
            )
            outputs.append(result[0]["generated_text"].strip())
            
        return outputs
    
    def save_results(
        self, 
        results: Dict[str, float], 
        output_path: Optional[str] = None
    ) -> str:
        """
        Save evaluation results to disk.
        
        Args:
            results: Dictionary of evaluation results.
            output_path: Path where to save the results.
            
        Returns:
            Path where the results were saved.
        """
        if output_path is None:
            output_dir = get_config_value(
                self.evaluation_config, "output_dir", "./evaluation_results"
            )
            model_name = get_config_value(
                self.model_config, "base_model_id", "model"
            ).split("/")[-1]
            output_path = os.path.join(output_dir, f"{model_name}_results.txt")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            for metric, value in results.items():
                f.write(f"{metric}: {value}\n")
                
        return output_path