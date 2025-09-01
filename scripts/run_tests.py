#!/usr/bin/env python
"""
Test runner script for the LLM Trainer project.

This script provides a convenient way to run tests with various options.
"""

import argparse
import subprocess
import sys


def run_command(command):
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for the LLM Trainer project")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests with verbose output")
    parser.add_argument("--file", type=str, help="Run specific test file")
    
    args = parser.parse_args()
    
    # Build pytest command
    command = ["python", "-m", "pytest"]
    
    if args.coverage:
        command.extend(["--cov=src", "--cov-report=term-missing"])
    
    if args.unit:
        command.extend(["-m", "unit"])
    elif args.integration:
        command.extend(["-m", "integration"])
    
    if args.verbose:
        command.append("-v")
    
    if args.file:
        command.append(f"tests/{args.file}")
    
    # If no specific options, run all tests
    if not any([args.unit, args.integration, args.file]):
        command.append("tests/")
    
    # Run the tests
    exit_code = run_command(command)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()