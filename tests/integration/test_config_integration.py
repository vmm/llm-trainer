"""Integration tests for core functionality."""

import unittest
from unittest.mock import Mock, patch

from src.utils.config import load_config, get_config_value


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration system."""

    def test_config_workflow(self):
        """Test a complete configuration workflow."""
        # Create a test config
        test_config = {
            "model": {
                "base_model_id": "test-model",
                "quantization": {
                    "load_in_4bit": True
                }
            },
            "training": {
                "num_epochs": 5,
                "learning_rate": 1e-4
            }
        }
        
        # Test nested value retrieval
        model_id = get_config_value(test_config, "model.base_model_id")
        self.assertEqual(model_id, "test-model")
        
        # Test deep nested value retrieval
        quantization = get_config_value(test_config, "model.quantization.load_in_4bit")
        self.assertTrue(quantization)
        
        # Test default value
        missing_value = get_config_value(test_config, "missing.path", "default")
        self.assertEqual(missing_value, "default")


if __name__ == "__main__":
    unittest.main()