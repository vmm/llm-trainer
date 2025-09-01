#!/usr/bin/env python3
"""Test script to reproduce the psutil AttributeError issue."""

import sys
from unittest.mock import Mock

def test_original_code():
    """Test the original code with broken psutil."""
    print("Testing original code with broken psutil...")
    
    # Create a mock psutil module without virtual_memory
    mock_psutil = Mock()
    del mock_psutil.virtual_memory  # This will cause AttributeError
    
    sys.modules['psutil'] = mock_psutil
    
    # Test the original code path from base_trainer.py
    dataloader_num_workers = 4
    adjusted_workers = dataloader_num_workers
    
    if dataloader_num_workers > 1:
        try:
            import psutil
            # Check if we're in a memory-constrained environment
            total_memory = psutil.virtual_memory().total / (1024**3)  # in GB
            if total_memory < 16:  # Less than 16GB RAM
                # Use fewer workers for memory-constrained environments
                adjusted_workers = 1
                print(f"Memory-constrained environment detected ({total_memory:.1f}GB RAM). "
                      f"Reducing dataloader workers from {dataloader_num_workers} to {adjusted_workers}.")
        except:
            # If we can't check memory, default to safe setting
            adjusted_workers = 1
            print(f"Unable to check system memory. Reducing dataloader workers to {adjusted_workers} for stability.")
    
    print(f"Original code result - adjusted_workers: {adjusted_workers}")
    return adjusted_workers

def test_with_attribute_check():
    """Test with improved code that checks for attribute existence."""
    print("\nTesting improved code with attribute check...")
    
    # Create a mock psutil module without virtual_memory
    mock_psutil = Mock()
    del mock_psutil.virtual_memory  # This will cause AttributeError
    
    sys.modules['psutil'] = mock_psutil
    
    # Test the improved code path
    dataloader_num_workers = 4
    adjusted_workers = dataloader_num_workers
    
    if dataloader_num_workers > 1:
        try:
            import psutil
            # Check if psutil has virtual_memory attribute before using it
            if hasattr(psutil, 'virtual_memory'):
                total_memory = psutil.virtual_memory().total / (1024**3)  # in GB
                if total_memory < 16:  # Less than 16GB RAM
                    # Use fewer workers for memory-constrained environments
                    adjusted_workers = 1
                    print(f"Memory-constrained environment detected ({total_memory:.1f}GB RAM). "
                          f"Reducing dataloader workers from {dataloader_num_workers} to {adjusted_workers}.")
            else:
                # psutil doesn't have virtual_memory attribute
                adjusted_workers = 1
                print(f"psutil.virtual_memory not available. Reducing dataloader workers to {adjusted_workers} for stability.")
        except Exception as e:
            # If we can't check memory, default to safe setting
            adjusted_workers = 1
            print(f"Unable to check system memory ({type(e).__name__}: {e}). Reducing dataloader workers to {adjusted_workers} for stability.")
    
    print(f"Improved code result - adjusted_workers: {adjusted_workers}")
    return adjusted_workers

if __name__ == "__main__":
    # Test both approaches
    original_result = test_original_code()
    improved_result = test_with_attribute_check()
    
    print(f"\nResults:")
    print(f"Original code: {original_result}")
    print(f"Improved code: {improved_result}")
    print(f"Both approaches work correctly: {original_result == improved_result == 1}")