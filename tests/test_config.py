"""
Comprehensive test cases for configuration utility functions.

Tests cover normal usage, error handling, and edge cases for:
- load_config
- save_config  
- update_config
- get_config_value
"""

import os
import pytest
import yaml
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_config, save_config, update_config, get_config_value


class TestLoadConfig:
    """Test cases for load_config function."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML configuration file."""
        config_data = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "load_in_4bit": True
            },
            "training": {
                "num_train_epochs": 3,
                "learning_rate": 2.0e-4
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config == config_data
        assert loaded_config["model"]["base_model_id"] == "meta-llama/Meta-Llama-3-8B"
        assert loaded_config["training"]["num_train_epochs"] == 3

    def test_load_empty_config(self, tmp_path):
        """Test loading an empty YAML configuration file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config is None

    def test_load_config_with_null_values(self, tmp_path):
        """Test loading a config with null values."""
        config_data = {
            "model": {
                "adapter_name_or_path": None,
                "hub_model_id": None
            }
        }
        
        config_file = tmp_path / "null_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config["model"]["adapter_name_or_path"] is None
        assert loaded_config["model"]["hub_model_id"] is None

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading a non-existent configuration file."""
        non_existent_file = tmp_path / "non_existent.yaml"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(str(non_existent_file))
        
        assert "Configuration file not found" in str(exc_info.value)
        assert str(non_existent_file) in str(exc_info.value)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading a file with invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))

    def test_load_complex_nested_config(self, tmp_path):
        """Test loading a complex nested configuration."""
        config_data = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "config": {
                    "quantization": {
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": "float16"
                    }
                }
            },
            "dataset": {
                "preprocessing": {
                    "template": "{question}\\n\\nAnswer: {answer}",
                    "filters": ["remove_empty", "remove_duplicates"]
                }
            }
        }
        
        config_file = tmp_path / "complex_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config == config_data
        assert loaded_config["model"]["config"]["quantization"]["load_in_4bit"] is True
        assert len(loaded_config["dataset"]["preprocessing"]["filters"]) == 2


class TestSaveConfig:
    """Test cases for save_config function."""
    
    def test_save_simple_config(self, tmp_path):
        """Test saving a simple configuration dictionary."""
        config_data = {
            "model": {"base_model_id": "test-model"},
            "training": {"epochs": 5}
        }
        
        config_file = tmp_path / "saved_config.yaml"
        save_config(config_data, str(config_file))
        
        assert config_file.exists()
        
        # Verify the saved content
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == config_data

    def test_save_config_creates_directories(self, tmp_path):
        """Test that save_config creates directories if they don't exist."""
        config_data = {"test": "value"}
        nested_dir = tmp_path / "nested" / "path" / "to"
        config_file = nested_dir / "config.yaml"
        
        save_config(config_data, str(config_file))
        
        assert config_file.exists()
        assert nested_dir.exists()
        
        # Verify content
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == config_data

    def test_save_empty_config(self, tmp_path):
        """Test saving an empty configuration dictionary."""
        config_data = {}
        config_file = tmp_path / "empty_saved.yaml"
        
        save_config(config_data, str(config_file))
        
        assert config_file.exists()
        
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == {}

    def test_save_config_with_none_values(self, tmp_path):
        """Test saving a config with None values."""
        config_data = {
            "model": {
                "adapter_path": None,
                "hub_token": None
            },
            "training": {
                "push_to_hub": False
            }
        }
        
        config_file = tmp_path / "none_values.yaml"
        save_config(config_data, str(config_file))
        
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["model"]["adapter_path"] is None
        assert loaded_data["model"]["hub_token"] is None
        assert loaded_data["training"]["push_to_hub"] is False

    def test_save_complex_nested_config(self, tmp_path):
        """Test saving a complex nested configuration."""
        config_data = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True
                }
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj"]
            },
            "evaluation": {
                "metrics": ["accuracy", "f1"],
                "generate_kwargs": {
                    "max_new_tokens": 128,
                    "temperature": 0.7
                }
            }
        }
        
        config_file = tmp_path / "complex_saved.yaml"
        save_config(config_data, str(config_file))
        
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == config_data
        assert loaded_data["model"]["quantization_config"]["load_in_4bit"] is True
        assert len(loaded_data["lora"]["target_modules"]) == 3

    def test_save_config_overwrites_existing(self, tmp_path):
        """Test that save_config overwrites existing files."""
        config_file = tmp_path / "overwrite_test.yaml"
        
        # Save initial config
        initial_config = {"initial": "value"}
        save_config(initial_config, str(config_file))
        
        # Save new config (should overwrite)
        new_config = {"new": "value", "updated": True}
        save_config(new_config, str(config_file))
        
        # Verify new content
        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == new_config
        assert "initial" not in loaded_data


class TestUpdateConfig:
    """Test cases for update_config function."""
    
    def test_update_simple_config(self):
        """Test updating a simple configuration."""
        original_config = {
            "model": {"base_model_id": "old-model"},
            "training": {"epochs": 3}
        }
        
        updates = {
            "model": {"base_model_id": "new-model"},
            "training": {"epochs": 5}
        }
        
        updated_config = update_config(original_config, updates)
        
        assert updated_config["model"]["base_model_id"] == "new-model"
        assert updated_config["training"]["epochs"] == 5
        
        # Note: original config may be modified due to shallow copy behavior
        # This is the actual behavior of the current implementation
        assert updated_config is not original_config  # Different objects

    def test_update_nested_config(self):
        """Test updating nested configuration values."""
        original_config = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "quantization": {
                    "load_in_4bit": True,
                    "compute_dtype": "float16"
                }
            },
            "training": {"epochs": 3}
        }
        
        updates = {
            "model": {
                "quantization": {
                    "load_in_4bit": False
                }
            }
        }
        
        updated_config = update_config(original_config, updates)
        
        assert updated_config["model"]["quantization"]["load_in_4bit"] is False
        assert updated_config["model"]["quantization"]["compute_dtype"] == "float16"  # Should remain unchanged
        assert updated_config["model"]["base_model_id"] == "meta-llama/Meta-Llama-3-8B"  # Should remain unchanged

    def test_update_config_new_keys_not_allowed(self):
        """Test that new keys raise KeyError when allow_new_keys=False."""
        original_config = {
            "model": {"base_model_id": "test-model"}
        }
        
        updates = {
            "new_section": {"new_value": "test"}
        }
        
        with pytest.raises(KeyError) as exc_info:
            update_config(original_config, updates, allow_new_keys=False)
        
        assert "Key 'new_section' not found in original configuration" in str(exc_info.value)

    def test_update_config_new_keys_allowed(self):
        """Test that new keys are added when allow_new_keys=True."""
        original_config = {
            "model": {"base_model_id": "test-model"}
        }
        
        updates = {
            "new_section": {"new_value": "test"},
            "model": {"new_param": "added"}
        }
        
        updated_config = update_config(original_config, updates, allow_new_keys=True)
        
        assert updated_config["new_section"]["new_value"] == "test"
        assert updated_config["model"]["new_param"] == "added"
        assert updated_config["model"]["base_model_id"] == "test-model"

    def test_update_config_nested_new_keys_not_allowed(self):
        """Test that nested new keys raise KeyError when allow_new_keys=False."""
        original_config = {
            "model": {
                "base_model_id": "test-model",
                "quantization": {"load_in_4bit": True}
            }
        }
        
        updates = {
            "model": {
                "quantization": {"new_param": "value"}
            }
        }
        
        with pytest.raises(KeyError) as exc_info:
            update_config(original_config, updates, allow_new_keys=False)
        
        assert "Key 'new_param' not found in original configuration" in str(exc_info.value)

    def test_update_config_replace_dict_with_value(self):
        """Test replacing a dictionary with a simple value."""
        original_config = {
            "model": {
                "quantization": {
                    "load_in_4bit": True,
                    "compute_dtype": "float16"
                }
            }
        }
        
        updates = {
            "model": {
                "quantization": False  # Replace entire dict with boolean
            }
        }
        
        updated_config = update_config(original_config, updates)
        
        assert updated_config["model"]["quantization"] is False

    def test_update_empty_config(self):
        """Test updating an empty configuration."""
        original_config = {}
        updates = {"new_key": "new_value"}
        
        updated_config = update_config(original_config, updates, allow_new_keys=True)
        
        assert updated_config["new_key"] == "new_value"

    def test_update_with_empty_updates(self):
        """Test updating with empty updates dictionary."""
        original_config = {
            "model": {"base_model_id": "test-model"},
            "training": {"epochs": 3}
        }
        
        updates = {}
        
        updated_config = update_config(original_config, updates)
        
        assert updated_config == original_config

    def test_update_config_shallow_copy_behavior(self):
        """Test that update_config uses shallow copy (current implementation behavior)."""
        import copy
        original_config = {
            "model": {"base_model_id": "old-model"},
            "training": {"epochs": 3}
        }
        
        # Make a deep copy to test against
        original_copy = copy.deepcopy(original_config)
        
        updates = {
            "model": {"base_model_id": "new-model"}
        }
        
        updated_config = update_config(original_config, updates)
        
        # The function creates a new top-level dict
        assert updated_config is not original_config
        
        # But nested dicts may be shared due to shallow copy
        # This documents the current behavior
        assert updated_config["model"]["base_model_id"] == "new-model"


class TestGetConfigValue:
    """Test cases for get_config_value function."""
    
    def test_get_simple_value(self):
        """Test getting a simple top-level value."""
        config = {
            "model_id": "test-model",
            "epochs": 5
        }
        
        assert get_config_value(config, "model_id") == "test-model"
        assert get_config_value(config, "epochs") == 5

    def test_get_nested_value(self):
        """Test getting a nested value using dot notation."""
        config = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "quantization": {
                    "load_in_4bit": True,
                    "compute_dtype": "float16"
                }
            },
            "training": {
                "optimizer": {
                    "name": "adamw",
                    "lr": 2e-4
                }
            }
        }
        
        assert get_config_value(config, "model.base_model_id") == "meta-llama/Meta-Llama-3-8B"
        assert get_config_value(config, "model.quantization.load_in_4bit") is True
        assert get_config_value(config, "training.optimizer.name") == "adamw"
        assert get_config_value(config, "training.optimizer.lr") == 2e-4

    def test_get_value_with_default(self):
        """Test getting a value with default when key doesn't exist."""
        config = {
            "model": {"base_model_id": "test-model"}
        }
        
        assert get_config_value(config, "nonexistent", "default_value") == "default_value"
        assert get_config_value(config, "model.nonexistent", 42) == 42
        assert get_config_value(config, "model.nested.deep", None) is None

    def test_get_value_no_default(self):
        """Test getting a non-existent value without default returns None."""
        config = {
            "model": {"base_model_id": "test-model"}
        }
        
        assert get_config_value(config, "nonexistent") is None
        assert get_config_value(config, "model.nonexistent") is None

    def test_get_value_empty_path(self):
        """Test getting value with empty path."""
        config = {
            "model": {"base_model_id": "test-model"}
        }
        
        # Empty string splits to [''] and tries to find key '' which doesn't exist
        assert get_config_value(config, "", "default") == "default"
        assert get_config_value(config, "") is None

    def test_get_value_from_empty_config(self):
        """Test getting value from empty configuration."""
        config = {}
        
        assert get_config_value(config, "any.path", "default") == "default"
        assert get_config_value(config, "any.path") is None

    def test_get_value_path_through_non_dict(self):
        """Test getting value when path goes through non-dictionary value."""
        config = {
            "model": {
                "base_model_id": "test-model",
                "epochs": 5  # This is not a dict
            }
        }
        
        # Trying to traverse through 'epochs' (which is an int) should return default
        assert get_config_value(config, "model.epochs.invalid", "default") == "default"
        assert get_config_value(config, "model.epochs.invalid") is None

    def test_get_value_with_none_values(self):
        """Test getting None values from configuration."""
        config = {
            "model": {
                "adapter_path": None,
                "hub_token": None
            }
        }
        
        assert get_config_value(config, "model.adapter_path") is None
        assert get_config_value(config, "model.hub_token") is None
        assert get_config_value(config, "model.adapter_path", "default") is None

    def test_get_value_deep_nesting(self):
        """Test getting values from deeply nested configuration."""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "deep_value": "found it!"
                            }
                        }
                    }
                }
            }
        }
        
        assert get_config_value(config, "level1.level2.level3.level4.level5.deep_value") == "found it!"
        assert get_config_value(config, "level1.level2.level3.level4.level5.nonexistent", "default") == "default"

    def test_get_value_with_list_values(self):
        """Test getting list values from configuration."""
        config = {
            "lora": {
                "target_modules": ["q_proj", "k_proj", "v_proj"],
                "config": {
                    "metrics": ["accuracy", "f1", "bleu"]
                }
            }
        }
        
        target_modules = get_config_value(config, "lora.target_modules")
        assert target_modules == ["q_proj", "k_proj", "v_proj"]
        assert len(target_modules) == 3
        
        metrics = get_config_value(config, "lora.config.metrics")
        assert metrics == ["accuracy", "f1", "bleu"]

    def test_get_value_with_mixed_types(self):
        """Test getting values of various types from configuration."""
        config = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
            "none_val": None
        }
        
        assert get_config_value(config, "string_val") == "test"
        assert get_config_value(config, "int_val") == 42
        assert get_config_value(config, "float_val") == 3.14
        assert get_config_value(config, "bool_val") is True
        assert get_config_value(config, "list_val") == [1, 2, 3]
        assert get_config_value(config, "dict_val") == {"nested": "value"}
        assert get_config_value(config, "none_val") is None


class TestConfigEdgeCases:
    """Additional edge case tests for comprehensive coverage."""
    
    def test_load_config_with_special_yaml_features(self, tmp_path):
        """Test loading config with special YAML features like anchors and references."""
        yaml_content = """
defaults: &defaults
  num_train_epochs: 3
  learning_rate: 2.0e-4

model:
  base_model_id: "test-model"
  
training:
  <<: *defaults
  per_device_train_batch_size: 4
"""
        config_file = tmp_path / "special_yaml.yaml"
        config_file.write_text(yaml_content)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config["training"]["num_train_epochs"] == 3
        assert loaded_config["training"]["learning_rate"] == 2.0e-4
        assert loaded_config["training"]["per_device_train_batch_size"] == 4

    def test_save_config_with_unicode_and_special_chars(self, tmp_path):
        """Test saving config with unicode characters and special strings."""
        config_data = {
            "model": {
                "name": "测试模型",  # Chinese characters
                "description": "Model with émojis 🚀 and special chars: ñáéíóú",
                "path": "/path/with spaces/and-dashes_underscores"
            },
            "special_values": {
                "multiline": "Line 1\nLine 2\nLine 3",
                "single_quote": "It's a test",
                "double_quote": 'He said "Hello"',
                "mixed": "Mix 'single' and \"double\" quotes"
            }
        }
        
        config_file = tmp_path / "unicode_config.yaml"
        save_config(config_data, str(config_file))
        
        # Verify it can be loaded back correctly
        loaded_config = load_config(str(config_file))
        assert loaded_config == config_data

    def test_update_config_with_complex_nested_structures(self):
        """Test updating config with lists, mixed types, and deep nesting."""
        original_config = {
            "model": {
                "target_modules": ["q_proj", "k_proj"],
                "config": {
                    "quantization": {
                        "bits": 4,
                        "enabled": True
                    }
                }
            },
            "training": {
                "metrics": ["accuracy", "f1"],
                "hyperparams": {
                    "lr_schedule": {
                        "type": "cosine",
                        "params": {"min_lr": 1e-6}
                    }
                }
            }
        }
        
        updates = {
            "model": {
                "target_modules": ["q_proj", "k_proj", "v_proj"],  # Replace entire list
                "config": {
                    "quantization": {
                        "bits": 8  # Update nested value
                    }
                }
            }
        }
        
        updated_config = update_config(original_config, updates)
        
        assert len(updated_config["model"]["target_modules"]) == 3
        assert "v_proj" in updated_config["model"]["target_modules"]
        assert updated_config["model"]["config"]["quantization"]["bits"] == 8
        assert updated_config["model"]["config"]["quantization"]["enabled"] is True  # Should remain

    def test_get_config_value_with_numeric_keys(self):
        """Test get_config_value with configurations containing numeric keys."""
        config = {
            "layers": {
                "0": {"type": "embedding", "size": 768},
                "1": {"type": "attention", "heads": 12},
                "12": {"type": "output", "vocab_size": 50000}
            },
            "model": {
                "version": "1.0",
                "layers": 13
            }
        }
        
        assert get_config_value(config, "layers.0.type") == "embedding"
        assert get_config_value(config, "layers.1.heads") == 12
        assert get_config_value(config, "layers.12.vocab_size") == 50000
        assert get_config_value(config, "model.version") == "1.0"

    def test_get_config_value_with_boolean_and_none_defaults(self):
        """Test get_config_value with various default value types."""
        config = {"existing": {"value": "found"}}
        
        # Test with different default types
        assert get_config_value(config, "missing", False) is False
        assert get_config_value(config, "missing", True) is True
        assert get_config_value(config, "missing", 0) == 0
        assert get_config_value(config, "missing", []) == []
        assert get_config_value(config, "missing", {}) == {}
        assert get_config_value(config, "missing", "default") == "default"

    def test_save_config_empty_nested_directories(self, tmp_path):
        """Test saving to deeply nested directory that gets created."""
        config_data = {"test": "value"}
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e" / "config.yaml"
        
        save_config(config_data, str(deep_path))
        
        assert deep_path.exists()
        assert deep_path.parent.exists()
        
        loaded_config = load_config(str(deep_path))
        assert loaded_config == config_data

    def test_load_config_permission_error_simulation(self, tmp_path):
        """Test load_config behavior when file exists but can't be read."""
        config_file = tmp_path / "readonly_config.yaml"
        config_file.write_text("test: value")
        
        # Make file unreadable (this might not work in all environments)
        try:
            config_file.chmod(0o000)
            
            # This should raise a PermissionError or similar
            with pytest.raises((PermissionError, OSError)):
                load_config(str(config_file))
        except (OSError, PermissionError):
            # If we can't change permissions, skip this test
            pytest.skip("Cannot modify file permissions in this environment")
        finally:
            # Restore permissions for cleanup
            try:
                config_file.chmod(0o644)
            except:
                pass

    def test_update_config_with_empty_nested_dicts(self):
        """Test update_config behavior with empty nested dictionaries."""
        original_config = {
            "model": {},
            "training": {"epochs": 3},
            "empty": {}
        }
        
        updates = {
            "model": {"new_param": "value"},
            "empty": {"now_has_value": True}
        }
        
        updated_config = update_config(original_config, updates, allow_new_keys=True)
        
        assert updated_config["model"]["new_param"] == "value"
        assert updated_config["empty"]["now_has_value"] is True
        assert updated_config["training"]["epochs"] == 3

    def test_config_functions_with_very_large_config(self, tmp_path):
        """Test config functions with a large configuration to ensure performance."""
        # Create a large nested config
        large_config = {}
        for i in range(100):
            large_config[f"section_{i}"] = {}
            for j in range(50):
                large_config[f"section_{i}"][f"param_{j}"] = {
                    "value": f"value_{i}_{j}",
                    "metadata": {
                        "type": "string",
                        "description": f"Parameter {j} in section {i}"
                    }
                }
        
        # Test save/load roundtrip
        config_file = tmp_path / "large_config.yaml"
        save_config(large_config, str(config_file))
        loaded_config = load_config(str(config_file))
        
        assert loaded_config == large_config
        
        # Test get_config_value on deep path
        value = get_config_value(loaded_config, "section_50.param_25.metadata.type")
        assert value == "string"
        
        # Test update_config
        updates = {"section_0": {"param_0": {"value": "updated"}}}
        updated_config = update_config(loaded_config, updates)
        assert updated_config["section_0"]["param_0"]["value"] == "updated"


class TestConfigIntegration:
    """Integration tests combining multiple config utility functions."""
    
    def test_save_load_roundtrip(self, tmp_path):
        """Test that saving and loading a config produces the same result."""
        original_config = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "load_in_4bit": True,
                "adapter_path": None
            },
            "training": {
                "num_train_epochs": 3,
                "learning_rate": 2.0e-4,
                "optimizer": {
                    "name": "adamw",
                    "params": {"weight_decay": 0.01}
                }
            },
            "lora": {
                "target_modules": ["q_proj", "k_proj", "v_proj"]
            }
        }
        
        config_file = tmp_path / "roundtrip_config.yaml"
        
        # Save and load
        save_config(original_config, str(config_file))
        loaded_config = load_config(str(config_file))
        
        assert loaded_config == original_config

    def test_update_and_save_workflow(self, tmp_path):
        """Test a typical workflow of loading, updating, and saving config."""
        # Create initial config
        initial_config = {
            "model": {
                "base_model_id": "old-model",
                "load_in_4bit": False
            },
            "training": {"epochs": 1}
        }
        
        config_file = tmp_path / "workflow_config.yaml"
        save_config(initial_config, str(config_file))
        
        # Load, update, and save
        loaded_config = load_config(str(config_file))
        
        updates = {
            "model": {
                "base_model_id": "new-model",
                "load_in_4bit": True
            },
            "training": {"epochs": 5}
        }
        
        updated_config = update_config(loaded_config, updates)
        save_config(updated_config, str(config_file))
        
        # Verify final result
        final_config = load_config(str(config_file))
        
        assert final_config["model"]["base_model_id"] == "new-model"
        assert final_config["model"]["load_in_4bit"] is True
        assert final_config["training"]["epochs"] == 5

    def test_get_config_values_from_loaded_config(self, tmp_path):
        """Test getting config values from a loaded configuration."""
        config_data = {
            "model": {
                "base_model_id": "meta-llama/Meta-Llama-3-8B",
                "quantization": {
                    "load_in_4bit": True,
                    "compute_dtype": "float16"
                }
            },
            "training": {
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 2e-4
                }
            }
        }
        
        config_file = tmp_path / "integration_config.yaml"
        save_config(config_data, str(config_file))
        loaded_config = load_config(str(config_file))
        
        # Test getting various values
        assert get_config_value(loaded_config, "model.base_model_id") == "meta-llama/Meta-Llama-3-8B"
        assert get_config_value(loaded_config, "model.quantization.load_in_4bit") is True
        assert get_config_value(loaded_config, "training.optimizer.name") == "adamw"
        assert get_config_value(loaded_config, "training.optimizer.learning_rate") == 2e-4
        assert get_config_value(loaded_config, "nonexistent.path", "default") == "default"