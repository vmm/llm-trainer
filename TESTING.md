# Testing Infrastructure

This document describes the testing infrastructure that has been added to the LLM Trainer project.

## Overview

The testing infrastructure provides comprehensive test coverage for configuration utilities and project structure validation. It includes both unit tests (fast, isolated) and integration tests (project-wide validation).

## Test Organization

### Test Structure
```
tests/
├── __init__.py                 # Package initialization
├── test_config.py             # Unit tests for configuration utilities
└── test_integration.py        # Integration tests for project structure
```

### Test Categories

#### Unit Tests (`@pytest.mark.unit`)
- **test_config.py**: Tests for configuration utilities in `src.utils.config`
  - Loading/saving YAML configuration files
  - Getting nested configuration values with dot notation
  - Updating configuration dictionaries
  - Error handling for invalid paths and files

#### Integration Tests (`@pytest.mark.integration`)
- **test_integration.py**: Tests for project structure and actual config files
  - Validation that required configuration files exist
  - Structure validation for actual config files (llama3_reasoning.yaml, gemma_tinystories.yaml)
  - Module import verification
  - Project directory structure validation

## Configuration Files

### pytest.ini
Basic pytest configuration for test discovery and output formatting.

### pyproject.toml
Modern Python project configuration with pytest markers definition.

### requirements-dev.txt
Development dependencies including:
- pytest>=7.0.0
- pytest-cov>=4.0.0 (for coverage reports)
- pytest-mock>=3.10.0 (for mocking support)

## Running Tests

### Using pytest directly
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific test categories
python -m pytest -m unit
python -m pytest -m integration

# Run specific test file
python -m pytest tests/test_config.py
```

### Using the test runner script
```bash
# Run all tests
python scripts/run_tests.py

# Run with coverage
python scripts/run_tests.py --coverage

# Run specific categories
python scripts/run_tests.py --unit
python scripts/run_tests.py --integration

# Run specific file
python scripts/run_tests.py --file test_config.py
```

## Test Coverage

The current test suite provides:
- **100% coverage** of the configuration utilities module
- **Comprehensive validation** of project structure and configuration files
- **Error handling tests** for invalid inputs and missing files
- **Integration validation** ensuring the project components work together

## Adding New Tests

When adding new tests:

1. **Unit tests**: Add to appropriate test file in `tests/`
2. **Mark tests appropriately**: Use `@pytest.mark.unit` or `@pytest.mark.integration`
3. **Follow naming conventions**: `test_*` for test functions, `Test*` for test classes
4. **Include docstrings**: Document what each test validates
5. **Test edge cases**: Include error conditions and boundary cases

## Benefits

This testing infrastructure provides:
- **Confidence in refactoring**: Tests catch regressions early
- **Documentation**: Tests serve as examples of how components should work
- **Quality assurance**: Automated validation of core functionality
- **Development speed**: Quick feedback on changes through fast unit tests
- **Integration validation**: Ensures project structure remains consistent