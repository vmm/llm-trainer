"""Test configuration utilities."""

import os
import tempfile
import pytest
import yaml
from src.utils.config import load_config, save_config, update_config, get_config_value


@pytest.mark.unit
class TestConfigUtils:
    """Test cases for configuration utilities."""
    
    def test_load_config_valid_file(self):
        """Test loading a valid YAML configuration file."""
        test_config = {
            "model": {
                "base_model_id": "test-model",
                "load_in_4bit": True
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 16
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == test_config
        finally:
            os.unlink(temp_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading a nonexistent configuration file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        test_config = {
            "test_key": "test_value",
            "nested": {
                "key": 42
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(test_config, temp_path)
            
            # Verify the file was created and contains correct content
            assert os.path.exists(temp_path)
            loaded_config = load_config(temp_path)
            assert loaded_config == test_config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_config_value_existing_path(self):
        """Test getting an existing value using dot notation."""
        config = {
            "model": {
                "base_model_id": "test-model",
                "parameters": {
                    "learning_rate": 0.001
                }
            },
            "training": {
                "epochs": 10
            }
        }
        
        assert get_config_value(config, "model.base_model_id") == "test-model"
        assert get_config_value(config, "model.parameters.learning_rate") == 0.001
        assert get_config_value(config, "training.epochs") == 10
    
    def test_get_config_value_nonexistent_path(self):
        """Test getting a nonexistent value returns default."""
        config = {
            "model": {
                "base_model_id": "test-model"
            }
        }
        
        assert get_config_value(config, "model.nonexistent") is None
        assert get_config_value(config, "model.nonexistent", "default") == "default"
        assert get_config_value(config, "nonexistent.path") is None
    
    def test_get_config_value_top_level(self):
        """Test getting top-level configuration values."""
        config = {
            "simple_key": "simple_value",
            "number": 42
        }
        
        assert get_config_value(config, "simple_key") == "simple_value"
        assert get_config_value(config, "number") == 42
    
    def test_update_config_existing_keys(self):
        """Test updating existing configuration keys."""
        original_config = {
            "model": {
                "base_model_id": "old-model",
                "load_in_4bit": True
            },
            "training": {
                "learning_rate": 0.001
            }
        }
        
        updates = {
            "model": {
                "base_model_id": "new-model"
            },
            "training": {
                "learning_rate": 0.002
            }
        }
        
        updated_config = update_config(original_config, updates)
        
        assert updated_config["model"]["base_model_id"] == "new-model"
        assert updated_config["model"]["load_in_4bit"] is True  # Should be preserved
        assert updated_config["training"]["learning_rate"] == 0.002
    
    def test_update_config_new_keys_not_allowed(self):
        """Test that adding new keys raises KeyError by default."""
        original_config = {
            "model": {
                "base_model_id": "test-model"
            }
        }
        
        updates = {
            "new_section": {
                "new_key": "new_value"
            }
        }
        
        with pytest.raises(KeyError):
            update_config(original_config, updates, allow_new_keys=False)
    
    def test_update_config_new_keys_allowed(self):
        """Test that adding new keys works when allowed."""
        original_config = {
            "model": {
                "base_model_id": "test-model"
            }
        }
        
        updates = {
            "new_section": {
                "new_key": "new_value"
            }
        }
        
        updated_config = update_config(original_config, updates, allow_new_keys=True)
        
        assert updated_config["model"]["base_model_id"] == "test-model"
        assert updated_config["new_section"]["new_key"] == "new_value"
    
    def test_get_config_value_empty_config(self):
        """Test getting values from an empty configuration."""
        config = {}
        
        assert get_config_value(config, "any.path") is None
        assert get_config_value(config, "any.path", "default") == "default"
    
    def test_get_config_value_none_values(self):
        """Test getting None values from configuration."""
        config = {
            "section": {
                "null_value": None,
                "empty_string": "",
                "zero": 0,
                "false_value": False
            }
        }
        
        # None values should be returned as None (not default)
        assert get_config_value(config, "section.null_value") is None
        assert get_config_value(config, "section.null_value", "default") is None
        
        # Other falsy values should be returned as-is
        assert get_config_value(config, "section.empty_string") == ""
        assert get_config_value(config, "section.zero") == 0
        assert get_config_value(config, "section.false_value") is False
    
    def test_save_config_creates_directory(self):
        """Test that save_config creates directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "subdir", "config.yaml")
            test_config = {"test": "value"}
            
            save_config(test_config, nested_path)
            
            assert os.path.exists(nested_path)
            loaded = load_config(nested_path)
            assert loaded == test_config