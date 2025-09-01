#!/usr/bin/env python3
"""Test runner script for the LLM Trainer project."""

import sys
import subprocess
from pathlib import Path


def run_tests(test_type=None, verbose=False, coverage=False):
    """Run the test suite.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Whether to run tests verbosely
        coverage: Whether to include coverage reporting
    """
    # Base command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test paths based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all" or test_type is None:
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM Trainer tests")
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests verbosely"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Include coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    if project_root.name == "scripts":
        project_root = project_root.parent
    
    import os
    os.chdir(project_root)
    
    # Run tests
    exit_code = run_tests(args.test_type, args.verbose, args.coverage)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()