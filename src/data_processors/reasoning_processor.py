"""Processor for reasoning datasets."""

import os
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict

from src.data_processors.base_processor import BaseDataProcessor
from src.utils.config import get_config_value


class ReasoningDataProcessor(BaseDataProcessor):
    """
    Processor for reasoning datasets like Natural Reasoning.
    
    This processor handles datasets containing reasoning problems with questions,
    context, and answers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reasoning data processor.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        
        # Get template for formatting the examples
        self.preprocessing = get_config_value(self.dataset_config, "preprocessing", {})
        self.template = get_config_value(
            self.preprocessing, "template", "{question}\n\nAnswer: {answer}"
        )
        
    def preprocess_dataset(self) -> DatasetDict:
        """
        Preprocess the Natural Reasoning dataset for instruction fine-tuning.
        
        The preprocessing steps include:
        1. Formatting the examples using the template
        2. Tokenizing the processed examples
        
        Returns:
            Processed dataset.
        """
        if self.raw_dataset is None:
            self.load_dataset()
            
        # Define the preprocessing function
        def format_examples(examples):
            """Format examples using the template."""
            formatted_texts = []
            
            for i in range(len(examples["question"])):
                question = examples["question"][i] if "question" in examples else ""
                answer = examples["answer"][i] if "answer" in examples else ""
                context = examples["context"][i] if "context" in examples else ""
                
                # Replace placeholders in the template
                text = self.template.format(
                    question=question,
                    answer=answer,
                    context=context
                )
                
                formatted_texts.append(text)
                
            return {self.text_column: formatted_texts}
        
        # Apply the format function to create the instruction dataset
        processed_dataset = self.raw_dataset.map(
            format_examples,
            batched=True,
            remove_columns=self.raw_dataset[self.train_split].column_names,
        )
        
        # Tokenize the dataset
        tokenized_dataset = {}
        for split in processed_dataset:
            tokenized_dataset[split] = self.tokenize_dataset(processed_dataset[split])
        
        self.processed_dataset = DatasetDict(tokenized_dataset)
        return self.processed_dataset


if __name__ == "__main__":
    """Run the processor as a standalone script."""
    import argparse
    from src.utils.config import load_config
    
    parser = argparse.ArgumentParser(description="Process a reasoning dataset for LLM fine-tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/llama3_reasoning.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where to save the processed dataset"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    processor = ReasoningDataProcessor(config)
    processed_dataset = processor.prepare_dataset()
    
    output_path = args.output_path
    if output_path:
        processor.save_processed_dataset(output_path)
        print(f"Processed dataset saved to {output_path}")
    else:
        dataset_name = get_config_value(config, "dataset.dataset_name", "").split("/")[-1]
        output_path = os.path.join("data", f"{dataset_name}_processed")
        print(f"Processed dataset saved to {output_path}")