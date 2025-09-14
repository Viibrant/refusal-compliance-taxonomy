# Rejection Detection

A Python package for detecting rejections in various contexts using natural language processing.

## Features

- Simple rejection detection based on keyword matching
- Extensible architecture for more sophisticated detection methods
- Comprehensive test coverage
- Type hints and documentation

## Installation

This project uses `uv` for dependency management. To install:

```bash
# Install the package in development mode
uv sync

# Install with test dependencies
uv sync --extra test

# Install with development dependencies
uv sync --extra dev
```

## Usage

### Basic Usage

```python
from rejection_detection import RejectionDetector

detector = RejectionDetector()

# Detect rejections in text
result = detector.detect("I reject this proposal")
print(result)  # True

result = detector.detect("This is an acceptance")
print(result)  # False
```

### Command Line Interface

```bash
# Run the CLI
uv run rejection-detection
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/rejection_detection --cov-report=html
```

### Code Formatting

```bash
# Format code with black
uv run black src tests

# Sort imports with isort
uv run isort src tests

# Lint with flake8
uv run flake8 src tests

# Type checking with mypy
uv run mypy src
```

## Project Structure

```
rejection-detection/
├── src/
│   └── rejection_detection/
│       └── __init__.py          # Main package code
├── tests/
│   ├── __init__.py
│   └── test_rejection_detection.py
├── pyproject.toml               # Project configuration
├── README.md
└── LICENSE
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.
