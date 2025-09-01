"""Unit tests for base evaluator."""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from src.evaluators.base_evaluator import BaseEvaluator


class MockEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""
    
    def load_dataset(self):
        """Mock implementation for testing."""
        self.dataset = Mock()
        return self.dataset
    
    def evaluate(self):
        """Mock implementation for testing."""
        self.metrics = {"accuracy": 0.85, "bleu": 0.72}
        return self.metrics


class TestBaseEvaluator(unittest.TestCase):
    """Test cases for BaseEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "model": {
                "base_model_id": "microsoft/DialoGPT-small",
                "trust_remote_code": True,
            },
            "evaluation": {
                "batch_size": 8,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": True,
            }
        }

    def test_init_sets_config_and_device(self):
        """Test that initialization sets config and device correctly."""
        evaluator = MockEvaluator(self.test_config)
        
        self.assertEqual(evaluator.config, self.test_config)
        self.assertEqual(evaluator.model_config, self.test_config["model"])
        self.assertEqual(evaluator.evaluation_config, self.test_config["evaluation"])
        
        # Device should be set (CPU since no GPU in CI)
        self.assertIsInstance(evaluator.device, torch.device)

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        minimal_config = {}
        evaluator = MockEvaluator(minimal_config)
        
        self.assertEqual(evaluator.config, minimal_config)
        self.assertEqual(evaluator.model_config, {})
        self.assertEqual(evaluator.evaluation_config, {})
        
        # Objects should be None initially
        self.assertIsNone(evaluator.model)
        self.assertIsNone(evaluator.tokenizer)
        self.assertIsNone(evaluator.pipeline)
        self.assertIsNone(evaluator.dataset)
        self.assertEqual(evaluator.metrics, {})

    @patch('src.evaluators.base_evaluator.AutoModelForCausalLM.from_pretrained')
    def test_load_model_with_default_path(self, mock_model_load):
        """Test loading model with default path from config."""
        mock_model = Mock()
        mock_model_load.return_value = mock_model
        
        evaluator = MockEvaluator(self.test_config)
        result = evaluator.load_model()
        
        # Verify model was loaded with correct parameters
        mock_model_load.assert_called_once_with(
            "microsoft/DialoGPT-small",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.assertEqual(result, mock_model)
        self.assertEqual(evaluator.model, mock_model)

    @patch('src.evaluators.base_evaluator.AutoModelForCausalLM.from_pretrained')
    def test_load_model_with_custom_path(self, mock_model_load):
        """Test loading model with custom path."""
        mock_model = Mock()
        mock_model_load.return_value = mock_model
        
        evaluator = MockEvaluator(self.test_config)
        custom_path = "/path/to/custom/model"
        result = evaluator.load_model(custom_path)
        
        # Verify model was loaded from custom path
        mock_model_load.assert_called_once_with(
            custom_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.assertEqual(result, mock_model)
        self.assertEqual(evaluator.model, mock_model)

    @patch('src.evaluators.base_evaluator.AutoTokenizer.from_pretrained')
    def test_load_tokenizer(self, mock_tokenizer_load):
        """Test loading tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_load.return_value = mock_tokenizer
        
        evaluator = MockEvaluator(self.test_config)
        result = evaluator.load_tokenizer()
        
        # Verify tokenizer was loaded
        mock_tokenizer_load.assert_called_once_with(
            "microsoft/DialoGPT-small",
            trust_remote_code=True
        )
        
        # Verify pad_token was set
        self.assertEqual(mock_tokenizer.pad_token, "<|endoftext|>")
        self.assertEqual(result, mock_tokenizer)
        self.assertEqual(evaluator.tokenizer, mock_tokenizer)

    @patch('src.evaluators.base_evaluator.pipeline')
    def test_setup_pipeline(self, mock_pipeline_func):
        """Test setting up inference pipeline."""
        mock_pipeline = Mock()
        mock_pipeline_func.return_value = mock_pipeline
        
        evaluator = MockEvaluator(self.test_config)
        evaluator.model = Mock()
        evaluator.tokenizer = Mock()
        
        result = evaluator.setup_pipeline()
        
        # Verify pipeline was created
        mock_pipeline_func.assert_called_once_with(
            "text-generation",
            model=evaluator.model,
            tokenizer=evaluator.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        self.assertEqual(result, mock_pipeline)
        self.assertEqual(evaluator.pipeline, mock_pipeline)

    def test_setup_pipeline_missing_model(self):
        """Test that setup_pipeline raises error when model is missing."""
        evaluator = MockEvaluator(self.test_config)
        
        with self.assertRaises(ValueError) as cm:
            evaluator.setup_pipeline()
        
        self.assertIn("Model must be loaded", str(cm.exception))

    def test_setup_pipeline_missing_tokenizer(self):
        """Test that setup_pipeline raises error when tokenizer is missing."""
        evaluator = MockEvaluator(self.test_config)
        evaluator.model = Mock()
        
        with self.assertRaises(ValueError) as cm:
            evaluator.setup_pipeline()
        
        self.assertIn("Tokenizer must be loaded", str(cm.exception))

    def test_mock_evaluator_methods(self):
        """Test that mock evaluator methods work as expected."""
        evaluator = MockEvaluator(self.test_config)
        
        # Test load_dataset mock
        dataset_result = evaluator.load_dataset()
        self.assertIsNotNone(dataset_result)
        self.assertEqual(evaluator.dataset, dataset_result)
        
        # Test evaluate mock
        metrics_result = evaluator.evaluate()
        expected_metrics = {"accuracy": 0.85, "bleu": 0.72}
        self.assertEqual(metrics_result, expected_metrics)
        self.assertEqual(evaluator.metrics, expected_metrics)

    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined correctly."""
        # This test ensures that BaseEvaluator cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseEvaluator(self.test_config)


if __name__ == "__main__":
    unittest.main()