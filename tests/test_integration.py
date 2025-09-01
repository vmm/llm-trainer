"""Integration tests for configuration and project structure."""

import os
import pytest
from src.utils.config import load_config, get_config_value


@pytest.mark.integration
class TestProjectIntegration:
    """Integration tests for the project structure and configurations."""
    
    def test_config_files_exist(self):
        """Test that required configuration files exist."""
        config_dir = "configs"
        assert os.path.exists(config_dir), "configs directory should exist"
        
        # Check for specific config files
        llama3_config = os.path.join(config_dir, "llama3_reasoning.yaml")
        gemma_config = os.path.join(config_dir, "gemma_tinystories.yaml")
        
        assert os.path.exists(llama3_config), "llama3_reasoning.yaml should exist"
        assert os.path.exists(gemma_config), "gemma_tinystories.yaml should exist"
    
    def test_llama3_config_structure(self):
        """Test that the llama3_reasoning.yaml has required structure."""
        config_path = "configs/llama3_reasoning.yaml"
        config = load_config(config_path)
        
        # Test required top-level sections
        required_sections = ["model", "dataset", "training", "lora", "evaluation"]
        for section in required_sections:
            assert section in config, f"Config should have '{section}' section"
        
        # Test model configuration
        assert get_config_value(config, "model.base_model_id") is not None
        assert get_config_value(config, "model.load_in_4bit") is not None
        
        # Test training configuration
        assert get_config_value(config, "training.output_dir") is not None
        assert get_config_value(config, "training.learning_rate") is not None
        assert get_config_value(config, "training.num_train_epochs") is not None
        
        # Test LoRA configuration
        assert get_config_value(config, "lora.r") is not None
        assert get_config_value(config, "lora.lora_alpha") is not None
        assert get_config_value(config, "lora.target_modules") is not None
    
    def test_gemma_config_structure(self):
        """Test that the gemma_tinystories.yaml has required structure."""
        config_path = "configs/gemma_tinystories.yaml"
        config = load_config(config_path)
        
        # Test required top-level sections
        required_sections = ["model", "dataset", "training", "lora"]
        for section in required_sections:
            assert section in config, f"Config should have '{section}' section"
        
        # Test model configuration
        assert get_config_value(config, "model.base_model_id") is not None
        
        # Test training configuration  
        assert get_config_value(config, "training.output_dir") is not None
        assert get_config_value(config, "training.learning_rate") is not None
    
    def test_src_modules_importable(self):
        """Test that key source modules can be imported."""
        # Test config utilities
        from src.utils.config import load_config, save_config, get_config_value, update_config
        
        # These should not raise ImportError
        assert callable(load_config)
        assert callable(save_config) 
        assert callable(get_config_value)
        assert callable(update_config)
    
    def test_project_structure(self):
        """Test that the project has the expected directory structure."""
        expected_dirs = [
            "src",
            "src/utils", 
            "src/trainers",
            "src/data_processors",
            "src/evaluators",
            "configs",
            "notebooks",
            "scripts"
        ]
        
        for dir_path in expected_dirs:
            assert os.path.exists(dir_path), f"Directory '{dir_path}' should exist"
            assert os.path.isdir(dir_path), f"'{dir_path}' should be a directory"
    
    def test_requirements_files_exist(self):
        """Test that requirements files exist."""
        assert os.path.exists("requirements.txt"), "requirements.txt should exist"
        assert os.path.exists("requirements-dev.txt"), "requirements-dev.txt should exist"
    
    @pytest.mark.integration
    def test_config_loading_integration(self):
        """Integration test for loading and using actual config files."""
        # Load the llama3 config
        config = load_config("configs/llama3_reasoning.yaml")
        
        # Test that we can access nested values
        model_id = get_config_value(config, "model.base_model_id")
        output_dir = get_config_value(config, "training.output_dir")
        learning_rate = get_config_value(config, "training.learning_rate")
        
        # Basic validation
        assert isinstance(model_id, str) and model_id
        assert isinstance(output_dir, str) and output_dir
        assert isinstance(learning_rate, float) and learning_rate > 0