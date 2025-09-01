#!/usr/bin/env python
"""
Test script to verify HuggingFace authentication functionality.
"""

import os
import sys
sys.path.append('.')

from src.utils.auth import (
    setup_huggingface_auth,
    get_auth_token,
    requires_authentication,
    validate_model_access
)


def test_auth_functions():
    """Test authentication functions."""
    
    print("Testing HuggingFace authentication functions...")
    print("=" * 50)
    
    # Test token retrieval
    print("1. Testing token retrieval:")
    token = get_auth_token()
    if token:
        print(f"   ✓ Found token: {token[:10]}..." if len(token) > 10 else f"   ✓ Found token: {token}")
    else:
        print("   - No token found (this is OK if HF_TOKEN is not set)")
    
    # Test authentication setup
    print("\n2. Testing authentication setup:")
    auth_result = setup_huggingface_auth()
    print(f"   Authentication result: {'✓ Success' if auth_result else '- No auth'}")
    
    # Test model requirement detection
    print("\n3. Testing gated model detection:")
    test_models = [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b-hf",
        "microsoft/DialoGPT-medium",
        "gpt2"
    ]
    
    for model in test_models:
        requires_auth = requires_authentication(model)
        print(f"   {model}: {'Requires auth' if requires_auth else 'No auth needed'}")
    
    # Test model access validation (without actually downloading)
    print("\n4. Testing model access validation:")
    for model in ["gpt2", "meta-llama/Meta-Llama-3-8B"]:
        try:
            accessible = validate_model_access(model)
            status = "✓ Accessible" if accessible else "✗ Not accessible"
            print(f"   {model}: {status}")
        except Exception as e:
            print(f"   {model}: Error checking access - {e}")
    
    print("\n" + "=" * 50)
    print("Authentication test completed!")
    
    # Provide guidance
    print("\nGuidance:")
    if not token:
        print("- To use gated models like Llama 3, set HF_TOKEN environment variable")
        print("- Get your token from https://huggingface.co/settings/tokens")
        print("- Example: export HF_TOKEN=your_token_here")
    else:
        print("- Token detected! You should be able to access gated models.")
    
    print("- For configuration files, add 'hf_token: your_token' under the model section")
    print("- Alternatively, set 'use_auth_token: false' to disable authentication")


if __name__ == "__main__":
    test_auth_functions()