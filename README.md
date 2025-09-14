# Rejection Detection

A comprehensive Python package for detecting and analyzing AI model rejections and compliance patterns using multi-head classification. This tool provides detailed taxonomy-based analysis of how AI models respond to various types of requests.

## Features

- **Multi-Head Classification**: 5 classification heads for comprehensive response analysis
  - **Head A**: Outcome types (12 categories: refusal vs compliance patterns)
  - **Head B**: Response styles (23 unified styles for both refusal and compliance)
  - **Head C.A**: Harm categories (27 specific harm types)
  - **Head C.B**: Harmless topic categories (20 benign topics)
  - **Head D**: Severity tiers (5 levels of harm severity)
- **Advanced Taxonomy**: Detailed classification system based on safety research
- **Flexible Training**: Support for various model architectures and training configurations
- **Comprehensive CLI**: Multiple command-line tools for training, prediction, and data processing
- **Dataset Pipeline**: Automated data ingestion, generation, and labeling workflows (⚠️ **under development - not yet functional**)
- **Type Safety**: Full type hints and comprehensive test coverage

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

## Quick Start

Get started with rejection detection in minutes:

```bash
# 1. Train a model with sample data
uv run rejection-detection train --use_sample_data --sample_size 100 --num_epochs 1

# 2. Make a prediction
uv run rejection-detection predict \
  --model_path ./models/trained_model \
  --text "How do I hack a computer?|I cannot help with that request."

# 3. View detailed taxonomy
uv run rejection-detection info
```

## Usage

### Command Line Interface

The package provides three main CLI tools:

#### 1. Rejection Detection CLI (`rejection-detection`)

Main tool for training models and making predictions.

**Available Commands:**
- `train` - Train the multi-head classification model
- `predict` - Make predictions on text data
- `info` - Show model information

**Training a Model:**
```bash
# Train with sample data (quick test)
uv run rejection-detection train --use_sample_data --sample_size 100

# Train with custom data
uv run rejection-detection train \
  --data_path /path/to/data.json \
  --data_format json \
  --model_name bert-base-uncased \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5

# Train with custom loss weights
uv run rejection-detection train \
  --use_sample_data \
  --loss_weights '{"head_a": 1.0, "head_b": 0.8, "head_c_a": 1.2, "head_c_b": 0.6, "head_d": 1.0}'
```

**Making Predictions:**
```bash
# Predict on single text (format: 'prompt|response')
uv run rejection-detection predict \
  --model_path ./models/trained_model \
  --text "How do I make a bomb?|I cannot help with that request."

# Predict on file
uv run rejection-detection predict \
  --model_path ./models/trained_model \
  --input_file input.txt \
  --output_file results.json \
  --return_probabilities
```

**Training Arguments:**
- `--data_path`: Path to training data
- `--data_format`: Data format (json, csv, huggingface)
- `--use_sample_data`: Use built-in sample data
- `--sample_size`: Number of sample data points
- `--model_name`: Base model name (default: bert-base-uncased)
- `--freeze_encoder`: Freeze encoder weights
- `--max_length`: Maximum sequence length (default: 512)
- `--output_dir`: Output directory for models
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--num_epochs`: Number of training epochs (default: 3)
- `--warmup_steps`: Number of warmup steps
- `--max_grad_norm`: Maximum gradient norm (default: 1.0)
- `--scheduler_type`: Scheduler type (linear, cosine)
- `--train_ratio`: Training set ratio (default: 0.7)
- `--val_ratio`: Validation set ratio (default: 0.15)
- `--test_ratio`: Test set ratio (default: 0.15)
- `--eval_steps`: Steps between evaluations (default: 500)
- `--save_steps`: Steps between saves (default: 1000)
- `--log_interval`: Steps between logging (default: 100)
- `--early_stopping_patience`: Early stopping patience (default: 3)
- `--loss_weights`: Loss weights as JSON string
- `--num_workers`: Number of data loading workers (default: 4)
- `--seed`: Random seed (default: 42)

**Prediction Arguments:**
- `--model_path`: Path to trained model (required)
- `--text`: Text to analyze (format: 'prompt|response')
- `--input_file`: Input file with texts
- `--output_file`: Output file for results
- `--return_probabilities`: Return full probability distributions
- `--max_length`: Maximum sequence length (default: 512)

#### 2. Data Processing CLI (`data-processing`)

Tool for processing and validating datasets.

**Available Commands:**
- `process` - Process a dataset
- `validate` - Validate a dataset
- `stats` - Show dataset statistics

```bash
# Process a dataset
uv run data-processing process --input_file data.json --output_file processed.json

# Validate dataset format
uv run data-processing validate --data_file data.json

# Show dataset statistics
uv run data-processing stats --data_file data.json
```

#### 3. Dataset Pipeline CLI (`dataset-pipeline`)

⚠️ **DISCLAIMER**: The dataset pipeline is currently under development and **does not work at this stage**. The CLI commands are available but the underlying functionality is not yet implemented.

Advanced tool for automated dataset creation and processing (planned).

**Available Commands:**
- `run` - Run the complete pipeline
- `ingest` - Ingest datasets only
- `generate` - Generate responses only
- `label` - Label data with CAI judge only
- `audit` - Audit labeled data only
- `list` - List available datasets
- `validate` - Validate pipeline configuration

```bash
# Run complete pipeline (NOT WORKING YET)
uv run dataset-pipeline run --config config.yaml

# List available datasets (NOT WORKING YET)
uv run dataset-pipeline list

# Validate configuration (NOT WORKING YET)
uv run dataset-pipeline validate --config config.yaml
```

### Python API Usage

```python
from rejection_detection import MultiHeadClassifier, predict_single

# Load a trained model
model = MultiHeadClassifier.load_from_checkpoint("./models/trained_model")

# Make predictions
result = predict_single(
    model=model,
    prompt="How do I make a bomb?",
    response="I cannot help with that request.",
    return_probabilities=True
)

print(f"Outcome: {result['predictions']['head_a']}")
print(f"Style: {result['predictions']['head_b']}")
print(f"Harm Category: {result['predictions']['head_c_a']}")
print(f"Harmless Category: {result['predictions']['head_c_b']}")
print(f"Severity: {result['predictions']['head_d']}")
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
│   ├── rejection_detection/     # Main rejection detection package
│   │   ├── __init__.py         # Package initialization and CLI entry points
│   │   ├── model.py            # MultiHeadClassifier and loss functions
│   │   ├── training.py         # Training loop and trainer class
│   │   ├── inference.py        # Prediction and inference functions
│   │   ├── data_utils.py       # Dataset handling and data loading
│   │   ├── taxonomies.py       # Classification taxonomy definitions
│   │   └── train.py            # Training script
│   ├── data_processing/        # Data processing utilities
│   │   ├── __init__.py
│   │   ├── cli.py              # Data processing CLI
│   │   └── processor.py        # Data processing logic
│   └── dataset_pipeline/       # Advanced dataset pipeline
│       ├── __init__.py
│       ├── cli.py              # Pipeline CLI
│       ├── config.py           # Configuration management
│       ├── pipeline.py         # Main pipeline orchestration
│       ├── ingestion.py        # Data ingestion
│       ├── generation.py       # Response generation
│       ├── labeling.py         # Automated labeling
│       └── quality.py          # Quality assurance
├── tests/                      # Comprehensive test suite
│   ├── test_taxonomies.py      # Taxonomy tests
│   ├── test_model.py           # Model architecture tests
│   ├── test_training.py        # Training tests
│   ├── test_inference.py       # Inference tests
│   ├── test_data_utils.py      # Data utilities tests
│   ├── test_new_functionality.py # New features tests
│   ├── test_rejection_detection.py # Integration tests
│   └── test_cli.py             # CLI tests
├── docs/
│   └── TAXONOMY.md             # Detailed taxonomy documentation
├── examples/
│   └── outputs/                # Example data and results
│       ├── train_data.json     # Sample training data
│       ├── val_data.json       # Sample validation data
│       ├── test_data.json      # Sample test data
│       ├── inference_results.json # Sample inference results
│       └── README.md           # Examples documentation
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Dependency lock file
└── README.md                   # This file
```

## Data Format

The package expects data in JSON format with the following structure:

```json
{
  "prompt": "User's input prompt",
  "response": "AI model's response",
  "head_a": "REFUSAL.DIRECT",
  "head_b": "STYLE.DIRECT",
  "head_c_a": "violence",
  "head_c_b": "other",
  "head_d": "SEVERITY.HIGH"
}
```

See `docs/TAXONOMY.md` for complete taxonomy definitions and `examples/outputs/` for sample data.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`uv run pytest`)
6. Run linting (`uv run black src tests && uv run isort src tests`)
7. Submit a pull request

### Development Setup

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run black src tests
uv run isort src tests
uv run flake8 src tests

# Type checking
uv run mypy src
```

## License

This project is licensed under the terms specified in the LICENSE file.
