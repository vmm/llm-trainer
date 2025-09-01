"""Unit tests for config utilities."""

import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import (
    load_config,
    save_config,
    update_config,
    get_config_value,
)


class TestConfigUtilities(unittest.TestCase):
    """Test cases for configuration utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "trust_remote_code": True,
                "quantization": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 4
            }
        }

    def test_get_config_value_existing_nested_path(self):
        """Test getting a value using a nested path that exists."""
        result = get_config_value(self.test_config, "model.base_model_id")
        self.assertEqual(result, "meta-llama/Meta-Llama-3-8B")

    def test_get_config_value_deep_nested_path(self):
        """Test getting a value using a deeply nested path."""
        result = get_config_value(self.test_config, "model.quantization.load_in_4bit")
        self.assertTrue(result)

    def test_get_config_value_nonexistent_path(self):
        """Test getting a value using a path that doesn't exist."""
        result = get_config_value(self.test_config, "nonexistent.path")
        self.assertIsNone(result)

    def test_get_config_value_with_default(self):
        """Test getting a value with a default for nonexistent path."""
        result = get_config_value(self.test_config, "nonexistent.path", "default_value")
        self.assertEqual(result, "default_value")

    def test_get_config_value_top_level(self):
        """Test getting a top-level value."""
        result = get_config_value(self.test_config, "model")
        expected = self.test_config["model"]
        self.assertEqual(result, expected)

    def test_update_config_basic(self):
        """Test basic config update."""
        updates = {"training": {"learning_rate": 1e-4}}
        result = update_config(self.test_config, updates, allow_new_keys=True)
        
        self.assertEqual(result["training"]["learning_rate"], 1e-4)
        self.assertEqual(result["training"]["num_epochs"], 3)  # Unchanged

    def test_update_config_new_key_not_allowed(self):
        """Test updating config with new key when not allowed."""
        updates = {"new_section": {"new_param": "value"}}
        
        with self.assertRaises(KeyError):
            update_config(self.test_config, updates, allow_new_keys=False)

    def test_update_config_new_key_allowed(self):
        """Test updating config with new key when allowed."""
        updates = {"new_section": {"new_param": "value"}}
        result = update_config(self.test_config, updates, allow_new_keys=True)
        
        self.assertEqual(result["new_section"]["new_param"], "value")

    def test_update_config_nested_update(self):
        """Test updating nested configuration values."""
        updates = {
            "model": {
                "quantization": {
                    "bnb_4bit_quant_type": "fp4"
                }
            }
        }
        result = update_config(self.test_config, updates, allow_new_keys=True)
        
        self.assertEqual(result["model"]["quantization"]["bnb_4bit_quant_type"], "fp4")
        self.assertTrue(result["model"]["quantization"]["load_in_4bit"])  # Unchanged

    def test_save_and_load_config(self):
        """Test saving and loading configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")
            
            # Save config
            save_config(self.test_config, config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            loaded_config = load_config(config_path)
            self.assertEqual(loaded_config, self.test_config)

    def test_load_config_file_not_found(self):
        """Test loading a config file that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_save_config_creates_directory(self):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nested", "dir", "config.yaml")
            
            save_config(self.test_config, config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Verify content
            loaded_config = load_config(config_path)
            self.assertEqual(loaded_config, self.test_config)


if __name__ == "__main__":
    unittest.main()