"""Unit tests for base data processor."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.data_processors.base_processor import BaseDataProcessor


class MockDataProcessor(BaseDataProcessor):
    """Concrete implementation of BaseDataProcessor for testing."""
    
    def preprocess_dataset(self):
        """Mock implementation for testing."""
        # Simple mock preprocessing
        self.processed_dataset = self.raw_dataset
        return self.processed_dataset


class TestBaseDataProcessor(unittest.TestCase):
    """Test cases for BaseDataProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "model": {
                "base_model_id": "microsoft/DialoGPT-small",  # Use a smaller model for testing
                "trust_remote_code": True,
            },
            "dataset": {
                "dataset_name": "test_dataset",
                "train_split": "train",
                "validation_split": "validation",
                "text_column": "text",
                "max_seq_length": 512,
            }
        }

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_init_with_default_values(self, mock_tokenizer):
        """Test initialization with default configuration values."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = {"model": {"base_model_id": "microsoft/DialoGPT-small"}}
        processor = MockDataProcessor(config)
        
        # Check default values are set
        self.assertEqual(processor.train_split, "train")
        self.assertEqual(processor.validation_split, "validation")
        self.assertEqual(processor.text_column, "text")
        self.assertEqual(processor.max_seq_length, 2048)
        self.assertIsNone(processor.dataset_name)

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_init_with_custom_values(self, mock_tokenizer):
        """Test initialization with custom configuration values."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        processor = MockDataProcessor(self.test_config)
        
        # Check custom values are set
        self.assertEqual(processor.train_split, "train")
        self.assertEqual(processor.validation_split, "validation")
        self.assertEqual(processor.text_column, "text")
        self.assertEqual(processor.max_seq_length, 512)
        self.assertEqual(processor.dataset_name, "test_dataset")

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_tokenizer_pad_token_setting(self, mock_tokenizer):
        """Test that pad_token is set when not available."""
        # Mock tokenizer without pad_token
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        processor = MockDataProcessor(self.test_config)
        
        # Verify pad_token was set to eos_token
        self.assertEqual(processor.tokenizer.pad_token, "<|endoftext|>")

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    @patch('src.data_processors.base_processor.datasets.load_dataset')
    def test_load_dataset_success(self, mock_load_dataset, mock_tokenizer):
        """Test successful dataset loading."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock dataset
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        processor = MockDataProcessor(self.test_config)
        result = processor.load_dataset()
        
        # Verify dataset was loaded
        mock_load_dataset.assert_called_once_with("test_dataset")
        self.assertEqual(result, mock_dataset)
        self.assertEqual(processor.raw_dataset, mock_dataset)

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_load_dataset_no_name(self, mock_tokenizer):
        """Test loading dataset without name raises error."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = {"model": {"base_model_id": "microsoft/DialoGPT-small"}}
        processor = MockDataProcessor(config)
        
        with self.assertRaises(ValueError) as cm:
            processor.load_dataset()
        
        self.assertIn("Dataset name must be specified", str(cm.exception))

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_tokenize_dataset(self, mock_tokenizer):
        """Test dataset tokenization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        # Mock tokenizer call
        def mock_tokenizer_call(texts, **kwargs):
            return {"input_ids": [[1, 2, 3] for _ in texts], "attention_mask": [[1, 1, 1] for _ in texts]}
        
        mock_tokenizer_instance.side_effect = mock_tokenizer_call
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.map = Mock(return_value=Mock())
        
        processor = MockDataProcessor(self.test_config)
        result = processor.tokenize_dataset(mock_dataset)
        
        # Verify map was called with correct parameters
        mock_dataset.map.assert_called_once()
        call_args = mock_dataset.map.call_args
        self.assertTrue(call_args[1]["batched"])
        self.assertEqual(call_args[1]["remove_columns"], ["label"])  # Should keep only text column

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_save_processed_dataset_error_when_not_processed(self, mock_tokenizer):
        """Test saving dataset when not processed raises error."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        processor = MockDataProcessor(self.test_config)
        
        with self.assertRaises(ValueError) as cm:
            processor.save_processed_dataset()
        
        self.assertIn("Dataset has not been processed yet", str(cm.exception))

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    def test_save_processed_dataset_with_custom_path(self, mock_tokenizer):
        """Test saving processed dataset with custom path."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        processor = MockDataProcessor(self.test_config)
        
        # Mock processed dataset
        mock_processed_dataset = Mock()
        processor.processed_dataset = mock_processed_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_dataset")
            result = processor.save_processed_dataset(custom_path)
            
            # Verify dataset save was called
            mock_processed_dataset.save_to_disk.assert_called_once_with(custom_path)
            self.assertEqual(result, custom_path)

    @patch('src.data_processors.base_processor.AutoTokenizer.from_pretrained')
    @patch('src.data_processors.base_processor.datasets.load_from_disk')
    def test_load_processed_dataset(self, mock_load_from_disk, mock_tokenizer):
        """Test loading processed dataset from disk."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock loaded dataset
        mock_dataset = Mock()
        mock_load_from_disk.return_value = mock_dataset
        
        processor = MockDataProcessor(self.test_config)
        result = processor.load_processed_dataset("/path/to/dataset")
        
        # Verify dataset was loaded
        mock_load_from_disk.assert_called_once_with("/path/to/dataset")
        self.assertEqual(result, mock_dataset)
        self.assertEqual(processor.processed_dataset, mock_dataset)


if __name__ == "__main__":
    unittest.main()