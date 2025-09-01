"""Authentication utilities for the LLM Trainer."""

import os
import warnings
from typing import Optional

try:
    from huggingface_hub import login, logout, whoami
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def setup_huggingface_auth(token: Optional[str] = None, use_auth_token: bool = True) -> bool:
    """
    Setup HuggingFace authentication.
    
    Args:
        token: HuggingFace token. If None, will look for HF_TOKEN environment variable.
        use_auth_token: Whether to use authentication token.
        
    Returns:
        True if authentication is successful, False otherwise.
    """
    if not use_auth_token:
        return True
        
    if not HF_HUB_AVAILABLE:
        warnings.warn(
            "huggingface_hub is not available. Install it with 'pip install huggingface_hub' "
            "to use authentication features."
        )
        return False
    
    # Get token from parameter or environment
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if token:
        try:
            login(token=token, add_to_git_credential=True)
            print("Successfully logged in to HuggingFace Hub")
            return True
        except Exception as e:
            warnings.warn(f"Failed to login to HuggingFace Hub: {e}")
            return False
    else:
        # Check if already logged in
        try:
            user_info = whoami()
            if user_info:
                print(f"Already logged in to HuggingFace Hub as: {user_info.get('name', 'unknown')}")
                return True
        except Exception:
            pass
            
        warnings.warn(
            "No HuggingFace token found. Some models (like Llama) may not be accessible. "
            "Set HF_TOKEN environment variable or provide token in config."
        )
        return False


def get_auth_token() -> Optional[str]:
    """
    Get HuggingFace authentication token.
    
    Returns:
        Authentication token if available, None otherwise.
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def validate_model_access(model_id: str, token: Optional[str] = None) -> bool:
    """
    Validate that we can access a model.
    
    Args:
        model_id: HuggingFace model ID to check.
        token: Optional token to use for validation.
        
    Returns:
        True if model is accessible, False otherwise.
    """
    if not HF_HUB_AVAILABLE:
        return True  # Assume accessible if we can't check
        
    try:
        from huggingface_hub import model_info
        
        # Try to get model info
        info = model_info(model_id, token=token or get_auth_token())
        return True
    except Exception as e:
        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            return False
        # For other errors, assume accessible
        return True


def requires_authentication(model_id: str) -> bool:
    """
    Check if a model requires authentication.
    
    Args:
        model_id: HuggingFace model ID to check.
        
    Returns:
        True if model likely requires authentication, False otherwise.
    """
    # Known gated models
    gated_models = [
        "meta-llama/Llama-2",
        "meta-llama/Meta-Llama-3",
        "meta-llama/CodeLlama",
    ]
    
    # Check if model ID starts with any known gated model prefix
    model_lower = model_id.lower()
    return any(gated.lower() in model_lower for gated in gated_models)