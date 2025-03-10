"""Configuration utilities for the LLM Trainer."""

import os
import yaml
from typing import Any, Dict, Optional, Union


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save a configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(
    config: Dict[str, Any], 
    updates: Dict[str, Any], 
    allow_new_keys: bool = False
) -> Dict[str, Any]:
    """
    Update a configuration dictionary with new values.
    
    Args:
        config: Original configuration dictionary.
        updates: Dictionary with updates to apply.
        allow_new_keys: Whether to allow adding new keys not present in the original config.
        
    Returns:
        Updated configuration dictionary.
    """
    updated_config = config.copy()
    
    def _update_nested_dict(d, u):
        for k, v in u.items():
            if k not in d and not allow_new_keys:
                raise KeyError(f"Key '{k}' not found in original configuration.")
            
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    return _update_nested_dict(updated_config, updates)


def get_config_value(
    config: Dict[str, Any], 
    key_path: str, 
    default: Optional[Any] = None
) -> Any:
    """
    Get a value from a nested configuration using a dot-separated path.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the desired value (e.g., "model.base_model_id").
        default: Default value to return if the path does not exist.
        
    Returns:
        The value at the specified path or the default value if not found.
    """
    keys = key_path.split(".")
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
            
    return result