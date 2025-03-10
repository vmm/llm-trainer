"""Evaluator for reasoning tasks."""

import os
import re
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import datasets

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from src.evaluators.base_evaluator import BaseEvaluator
from src.utils.config import get_config_value, load_config


class ReasoningEvaluator(BaseEvaluator):
    """
    Evaluator for reasoning tasks like LogiQA.
    
    This evaluator tests a model's ability to perform logical reasoning
    on multiple-choice reasoning questions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reasoning evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        
        # Get evaluation dataset name
        self.eval_dataset_name = get_config_value(
            self.evaluation_config, "eval_dataset", "logiqa"
        )
        
    def load_dataset(self, dataset_path: Optional[str] = None) -> Union[Dataset, DatasetDict]:
        """
        Load the evaluation dataset.
        
        Args:
            dataset_path: Optional path to a local dataset.
            
        Returns:
            Evaluation dataset.
        """
        if dataset_path and os.path.exists(dataset_path):
            # Load from local path
            self.dataset = datasets.load_from_disk(dataset_path)
        else:
            # Load from Hugging Face
            self.dataset = load_dataset(self.eval_dataset_name)
            
        return self.dataset
    
    def preprocess_question(self, example: Dict[str, Any]) -> str:
        """
        Format a reasoning question for the model.
        
        Args:
            example: Dataset example containing question and options.
            
        Returns:
            Formatted question string.
        """
        question = example.get("question", "")
        context = example.get("context", "")
        options = example.get("options", [])
        
        # Format the question
        prompt = ""
        if context:
            prompt += f"Context: {context}\n\n"
            
        prompt += f"Question: {question}\n\n"
        
        # Add options if available
        if options:
            prompt += "Options:\n"
            for i, option in enumerate(options):
                prompt += f"{chr(65 + i)}. {option}\n"
                
        prompt += "\nAnswer: "
        
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from the model's output.
        
        Args:
            text: Model's generated text.
            
        Returns:
            Extracted answer letter (A, B, C, D) or the raw text if no letter is found.
        """
        # Try to find an answer letter pattern
        pattern = r"^\s*([A-D])[\.:]"
        match = re.search(pattern, text)
        
        if match:
            return match.group(1)
        
        # Check for answer letter at the beginning of text
        if text and text[0] in "ABCD":
            return text[0]
        
        # Check for any letter in the text
        for char in text:
            if char in "ABCD":
                return char
                
        return text.strip()
    
    def evaluate(
        self, 
        model_path: Optional[str] = None, 
        dataset_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on reasoning tasks.
        
        Args:
            model_path: Path to the model or adapter.
            dataset_path: Path to the evaluation dataset.
            
        Returns:
            Dictionary of evaluation results.
        """
        # Load model if not already loaded
        if self.model is None and model_path is not None:
            self.load_model(model_path)
        elif self.model is None:
            self.load_model()
            
        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            self.load_tokenizer()
            
        # Set up pipeline
        if self.pipeline is None:
            self.setup_pipeline()
            
        # Load dataset
        if self.dataset is None:
            self.load_dataset(dataset_path)
            
        # Load metrics
        if not self.metrics:
            self.load_metrics()
            
        # Get test split
        test_dataset = self.dataset.get("test", self.dataset.get("validation", None))
        if test_dataset is None:
            raise ValueError("No test or validation split found in the dataset.")
            
        # Prepare questions
        questions = []
        answers = []
        
        for example in test_dataset:
            question_text = self.preprocess_question(example)
            questions.append(question_text)
            
            # Get ground truth answer
            if "answer" in example:
                answer = example["answer"]
                if isinstance(answer, int):
                    answer = chr(65 + answer)  # Convert 0->A, 1->B, etc.
                answers.append(answer)
            else:
                answers.append("")
                
        # Generate predictions
        print(f"Generating predictions for {len(questions)} questions...")
        predictions = []
        
        for i, question in enumerate(tqdm(questions)):
            gen_kwargs = get_config_value(self.evaluation_config, "generate_kwargs", {})
            result = self.pipeline(
                question,
                return_full_text=False,
                **gen_kwargs
            )
            generated_text = result[0]["generated_text"].strip()
            
            # Extract answer
            answer = self.extract_answer(generated_text)
            predictions.append(answer)
            
        # Compute metrics
        results = {}
        
        if "accuracy" in self.metrics and answers:
            # Convert string predictions to integer labels if needed
            pred_labels = []
            true_labels = []
            
            for pred, true in zip(predictions, answers):
                if pred in "ABCD" and true in "ABCD":
                    pred_labels.append(ord(pred) - ord('A'))
                    true_labels.append(ord(true) - ord('A'))
            
            # Compute accuracy
            if pred_labels:
                accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(pred_labels)
                results["accuracy"] = accuracy
        
        # Save detailed results
        output_dir = get_config_value(
            self.evaluation_config, "output_dir", "./evaluation_results"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed predictions
        with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
            for i, (question, pred, true) in enumerate(zip(questions, predictions, answers)):
                f.write(f"Question {i+1}:\n{question}\n\n")
                f.write(f"Predicted: {pred}\n")
                f.write(f"Ground truth: {true}\n")
                f.write("-" * 50 + "\n\n")
                
        # Print and save summary results
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
        self.save_results(results)
        
        return results


if __name__ == "__main__":
    """Run the evaluator as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on reasoning tasks")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/llama3_reasoning.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model or adapter to evaluate"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Update config with command line arguments
    if args.output_dir:
        if "evaluation" not in config:
            config["evaluation"] = {}
        config["evaluation"]["output_dir"] = args.output_dir
    
    evaluator = ReasoningEvaluator(config)
    results = evaluator.evaluate(args.model_path, args.dataset_path)
    
    print("Evaluation complete.")