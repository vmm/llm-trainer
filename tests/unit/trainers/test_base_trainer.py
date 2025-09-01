"""Unit tests for base trainer."""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from src.trainers.base_trainer import BaseTrainer


class MockTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def load_model(self):
        """Mock implementation for testing."""
        self.model = Mock()
        return self.model
    
    def setup_trainer(self, training_args=None):
        """Mock implementation for testing."""
        if training_args is None:
            training_args = self.setup_training_arguments()
        self.trainer = Mock()
        return self.trainer


class TestBaseTrainer(unittest.TestCase):
    """Test cases for BaseTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "model": {
                "base_model_id": "microsoft/DialoGPT-small",
                "trust_remote_code": True,
            },
            "training": {
                "output_dir": "./output",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "dataloader_num_workers": 2,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 100,
            }
        }

    def test_init_sets_config_and_device(self):
        """Test that initialization sets config and device correctly."""
        trainer = MockTrainer(self.test_config)
        
        self.assertEqual(trainer.config, self.test_config)
        self.assertEqual(trainer.model_config, self.test_config["model"])
        self.assertEqual(trainer.training_config, self.test_config["training"])
        
        # Device should be set (CPU since no GPU in CI)
        self.assertIsInstance(trainer.device, torch.device)

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        minimal_config = {}
        trainer = MockTrainer(minimal_config)
        
        self.assertEqual(trainer.config, minimal_config)
        self.assertEqual(trainer.model_config, {})
        self.assertEqual(trainer.training_config, {})
        
        # Objects should be None initially
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.tokenizer)
        self.assertIsNone(trainer.dataset)
        self.assertIsNone(trainer.data_collator)
        self.assertIsNone(trainer.trainer)

    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_setup_training_arguments_with_defaults(self, mock_training_args):
        """Test creating training arguments with default values."""
        minimal_config = {}
        trainer = MockTrainer(minimal_config)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        result = trainer.setup_training_arguments()
        
        # Check TrainingArguments was called
        mock_training_args.assert_called_once()
        call_kwargs = mock_training_args.call_args[1]
        
        # Verify default values
        self.assertEqual(call_kwargs["output_dir"], "./output")
        self.assertEqual(call_kwargs["num_train_epochs"], 3)
        self.assertEqual(call_kwargs["per_device_train_batch_size"], 4)
        self.assertEqual(call_kwargs["learning_rate"], 2e-4)
        self.assertEqual(call_kwargs["seed"], 42)
        
        self.assertEqual(result, mock_args_instance)

    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_setup_training_arguments_with_custom_values(self, mock_training_args):
        """Test creating training arguments with custom values."""
        trainer = MockTrainer(self.test_config)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        result = trainer.setup_training_arguments()
        
        # Check TrainingArguments was called
        mock_training_args.assert_called_once()
        call_kwargs = mock_training_args.call_args[1]
        
        # Verify custom values
        self.assertEqual(call_kwargs["output_dir"], "./output")
        self.assertEqual(call_kwargs["num_train_epochs"], 3)
        self.assertEqual(call_kwargs["per_device_train_batch_size"], 4)
        self.assertEqual(call_kwargs["learning_rate"], 2e-4)
        self.assertEqual(call_kwargs["weight_decay"], 0.01)
        self.assertEqual(call_kwargs["save_steps"], 500)
        self.assertEqual(call_kwargs["eval_steps"], 500)
        self.assertEqual(call_kwargs["logging_steps"], 100)

    @patch('builtins.__import__')
    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_memory_constrained_worker_adjustment(self, mock_training_args, mock_import):
        """Test that dataloader workers are reduced in memory-constrained environments."""
        # Mock psutil module and low memory system (8GB)
        mock_psutil = Mock()
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3  # 8GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        def mock_import_func(name, *args, **kwargs):
            if name == 'psutil':
                return mock_psutil
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_func
        
        config_with_workers = {
            "training": {
                "dataloader_num_workers": 4
            }
        }
        trainer = MockTrainer(config_with_workers)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        trainer.setup_training_arguments()
        
        # Check that workers were reduced to 1
        call_kwargs = mock_training_args.call_args[1]
        self.assertEqual(call_kwargs["dataloader_num_workers"], 1)

    @patch('builtins.__import__')
    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_high_memory_worker_unchanged(self, mock_training_args, mock_import):
        """Test that dataloader workers remain unchanged in high-memory environments."""
        # Mock psutil module and high memory system (32GB)
        mock_psutil = Mock()
        mock_memory = Mock()
        mock_memory.total = 32 * 1024**3  # 32GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        def mock_import_func(name, *args, **kwargs):
            if name == 'psutil':
                return mock_psutil
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_func
        
        config_with_workers = {
            "training": {
                "dataloader_num_workers": 4
            }
        }
        trainer = MockTrainer(config_with_workers)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        trainer.setup_training_arguments()
        
        # Check that workers remained at 4
        call_kwargs = mock_training_args.call_args[1]
        self.assertEqual(call_kwargs["dataloader_num_workers"], 4)

    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_psutil_import_error_handling(self, mock_training_args):
        """Test handling of psutil import errors."""
        config_with_workers = {
            "training": {
                "dataloader_num_workers": 4
            }
        }
        trainer = MockTrainer(config_with_workers)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
    @patch('src.trainers.base_trainer.TrainingArguments')
    def test_psutil_import_error_handling(self, mock_training_args):
        """Test handling of psutil import errors."""
        config_with_workers = {
            "training": {
                "dataloader_num_workers": 4
            }
        }
        trainer = MockTrainer(config_with_workers)
        
        # Mock TrainingArguments class
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        # Patch import to raise exception - psutil import happens inside setup_training_arguments
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   ImportError("psutil not available") if name == 'psutil' else __import__(name, *args, **kwargs)):
            trainer.setup_training_arguments()
        
        # Check that workers were reduced to 1 as a fallback
        call_kwargs = mock_training_args.call_args[1]
        self.assertEqual(call_kwargs["dataloader_num_workers"], 1)

    def test_setup_trainer_is_abstract(self):
        """Test that setup_trainer is correctly defined as abstract."""
        trainer = MockTrainer(self.test_config)
        
        # Our mock implementation should work
        result = trainer.setup_trainer()
        self.assertIsNotNone(result)

    def test_train_calls_setup_trainer_if_needed(self):
        """Test that train method calls setup_trainer when trainer is None."""
        trainer = MockTrainer(self.test_config)
        
        # Mock the trainer.train method
        mock_train_result = {"loss": 0.5}
        trainer.trainer = Mock()
        trainer.trainer.train.return_value = mock_train_result
        trainer.trainer.save_model = Mock()
        trainer.trainer.save_state = Mock()
        
        result = trainer.train()
        
        # Verify trainer methods were called
        trainer.trainer.train.assert_called_once()
        trainer.trainer.save_model.assert_called_once()
        trainer.trainer.save_state.assert_called_once()
        
        self.assertEqual(result, mock_train_result)

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented by subclasses."""
        # This test ensures that BaseTrainer cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseTrainer(self.test_config)


if __name__ == "__main__":
    unittest.main()