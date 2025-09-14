# Dataset Pipeline Module

A comprehensive pipeline for processing and labeling datasets for rejection detection model training. This module implements the end-to-end workflow for creating high-quality, multi-axis labeled datasets.

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
uv sync

# Option 1: Set up API keys (for commercial models)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Option 2: Use open source models (no API keys needed)
# Models will be downloaded automatically from Hugging Face
# See examples/opensource_models.md for recommended models
```

### 2. Basic Usage

```bash
# List available datasets
uv run dataset-pipeline list

# Run complete pipeline with default settings (uses open source models)
uv run dataset-pipeline run --output-dir outputs/my_dataset

# Run with specific datasets
uv run dataset-pipeline run --datasets wildguard_mix sorry_bench --output-dir outputs/custom

# Run with open source models configuration
uv run dataset-pipeline run --config examples/dataset_pipeline_config_opensource.json
```

## Step-by-Step Guide

### Step 1: Dataset Ingestion

**Purpose**: Load data from various sources (Hugging Face, GitHub, local files)

```bash
# Ingest all default datasets
uv run dataset-pipeline ingest --output-dir outputs/raw_data

# Ingest specific datasets
uv run dataset-pipeline ingest --datasets wildguard_mix sorry_bench --output-dir outputs/raw_data

# Check what was ingested
ls outputs/raw_data/
# You'll see: wildguard_mix_data.json, wildguard_mix_metadata.json, etc.
```

**Available Datasets**:
- `wildguard_mix` - WildGuardMix training data (86K+ items)
- `wildguard_test` - WildGuardMix test data (1.7K human-annotated)
- `sorry_bench` - SORRY-Bench (440 unsafe instructions)
- `do_not_answer` - Do-Not-Answer dataset
- `jailbreak_bench` - JailbreakBench (100 misuse behaviors)
- `or_bench` - OR-Bench (80K over-refusal prompts)

### Step 2: Response Generation (Optional)

**Purpose**: Generate responses for prompt-only datasets using multiple LLMs

```bash
# Generate responses for all ingested data
uv run dataset-pipeline generate --input-dir outputs/raw_data --output-dir outputs/generated

# Generate with commercial models
uv run dataset-pipeline generate \
  --input-dir outputs/raw_data \
  --output-dir outputs/generated \
  --models gpt-3.5-turbo claude-3-haiku \
  --temperature 0.7 \
  --max-tokens 512

# Generate with open source models
uv run dataset-pipeline generate \
  --input-dir outputs/raw_data \
  --output-dir outputs/generated \
  --models microsoft/DialoGPT-large mistralai/Mistral-7B-Instruct-v0.1 \
  --temperature 0.7 \
  --max-tokens 512
```

**Note**: This step is only needed for datasets that don't already have responses (like SORRY-Bench, Do-Not-Answer).

### Step 3: CAI Judge Labeling

**Purpose**: Label all data using Constitutional AI principles

```bash
# Label all data (ingested + generated) with commercial model
uv run dataset-pipeline label \
  --input-file outputs/processed/combined_data.json \
  --output-file outputs/labeled/labeling_results.json \
  --judge-model gpt-4 \
  --confidence-threshold 0.7 \
  --batch-size 10

# Label with open source model
uv run dataset-pipeline label \
  --input-file outputs/processed/combined_data.json \
  --output-file outputs/labeled/labeling_results.json \
  --judge-model microsoft/DialoGPT-large \
  --confidence-threshold 0.7 \
  --batch-size 10
```

**What gets labeled**:
- **Head A**: Outcome (REFUSAL.DIRECT, COMPLY.BENIGN, etc.)
- **Head B.A**: Refusal style (DIRECT, EXPLAIN, REDIRECT, etc.)
- **Head B.B**: Compliance style (DIRECT_ANSWER, STEP_BY_STEP, etc.)
- **Head C**: Harm categories (weapons, violence, self_harm, etc.)
- **Head D**: Safety flags (prompt_harmful, response_harmful, response_refusal)

### Step 4: Quality Control & Auditing

**Purpose**: Validate data quality and identify issues

```bash
# Audit labeled data
uv run dataset-pipeline audit \
  --input-file outputs/labeled/labeling_results.json \
  --output-dir outputs/audit \
  --human-sample-rate 0.05 \
  --min-confidence 0.6
```

**Audit includes**:
- PII detection (emails, phones, SSNs)
- Toxicity detection
- Duplicate identification
- Confidence analysis
- Coverage analysis
- Human review samples

### Step 5: Final Dataset Creation

**Purpose**: Combine all processed data into training-ready format

The pipeline automatically creates the final dataset in the processed directory:
- `final_dataset.json` - Complete multi-head labeled dataset
- `final_dataset.jsonl` - Line-delimited format for easy processing
- `dataset_info.json` - Dataset metadata and statistics

## Open Source Models

The pipeline supports using open source models from Hugging Face instead of commercial APIs. This is often more cost-effective and gives you full control over the models.

### Recommended Open Source Models

#### For Response Generation
- **Microsoft DialoGPT**: `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Zephyr**: `HuggingFaceH4/zephyr-7b-alpha`
- **LLaMA 2**: `meta-llama/Llama-2-7b-chat-hf` (requires approval)

#### For CAI Judge Labeling
- **Mistral Instruct**: `mistralai/Mistral-7B-Instruct-v0.1`
- **LLaMA 2 Chat**: `meta-llama/Llama-2-7b-chat-hf`
- **DialoGPT Large**: `microsoft/DialoGPT-large`

### Hardware Requirements

#### CPU-only Setup
- **RAM**: 8GB minimum, 16GB recommended
- **Models**: DialoGPT-medium, GPT-2 variants
- **Speed**: Slower but functional

#### Single GPU Setup
- **GPU**: 8GB VRAM minimum (RTX 3070, RTX 4060 Ti)
- **Models**: DialoGPT-large, smaller 7B models
- **Speed**: Good performance

#### Multi-GPU Setup
- **GPU**: 16GB+ VRAM per GPU (RTX 4080, RTX 4090, A100)
- **Models**: LLaMA 2, Mistral, Zephyr
- **Speed**: Excellent performance

### Usage Examples

```bash
# Use open source models for generation
uv run dataset-pipeline generate \
  --input-dir outputs/raw \
  --output-dir outputs/generated \
  --models microsoft/DialoGPT-large mistralai/Mistral-7B-Instruct-v0.1

# Use open source model for labeling
uv run dataset-pipeline label \
  --input-file data.json \
  --judge-model microsoft/DialoGPT-large \
  --output-file labeled.json

# Run complete pipeline with open source models
uv run dataset-pipeline run --config examples/dataset_pipeline_config_opensource.json
```

### Model Access

Most open source models are freely available, but some require approval:

```bash
# For LLaMA 2 models (requires approval)
huggingface-cli login

# For other models (no setup needed)
# Models download automatically
```

See `examples/opensource_models.md` for a comprehensive list of recommended models and configuration examples.

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```json
{
  "datasets": [
    {
      "name": "my_custom_dataset",
      "source_type": "local",
      "source_path": "data/my_data.json",
      "columns": {
        "prompt": "question",
        "response": "answer"
      }
    }
  ],
  "generation": {
    "models": ["gpt-4", "claude-3-opus"],
    "temperature": 0.5,
    "max_tokens": 1024
  },
  "labeling": {
    "judge_model": "gpt-4",
    "confidence_threshold": 0.8,
    "two_pass_evaluation": true
  },
  "quality": {
    "human_sample_rate": 0.1,
    "min_confidence": 0.7,
    "pii_scrubbing": true
  }
}
```

Run with custom config:

```bash
uv run dataset-pipeline run --config my_config.json
```

### Constitutional Rules

Create custom labeling rules:

```json
{
  "harm_categories": ["weapons", "violence", "self_harm", "hate_speech"],
  "refusal_indicators": ["I can't", "I won't", "not allowed"],
  "labeling_rules": {
    "outcome_determination": {
      "REFUSAL.DIRECT": "prompt_harmful=true AND response_refusal=true",
      "COMPLY.BENIGN": "prompt_harmful=false AND response_refusal=false"
    }
  }
}
```

Use with constitution file:

```bash
uv run dataset-pipeline run --config config.json --constitution constitution.json
```

### Programmatic Usage

```python
from dataset_pipeline import DatasetPipeline, PipelineConfig, DatasetSource

# Create configuration
config = PipelineConfig(
    output_dir=Path("outputs/my_dataset"),
    datasets=[
        DatasetSource(
            name="my_data",
            source_type="local",
            source_path="data.json"
        )
    ]
)

# Run pipeline
pipeline = DatasetPipeline(config)
result = pipeline.run_pipeline()

if result.success:
    print(f"Processed {result.processed_items} items")
```

## Output Structure

```
outputs/dataset_pipeline/
├── raw/                          # Step 1: Ingested data
│   ├── wildguard_mix_data.json
│   ├── wildguard_mix_metadata.json
│   └── wildguard_mix_provenance.json
├── processed/                    # Steps 2-3: Generated & labeled
│   ├── wildguard_mix_generated_responses.json
│   ├── labeling_results.json
│   ├── final_dataset.json       # Final training dataset
│   ├── final_dataset.jsonl      # Line-delimited format
│   └── dataset_info.json        # Dataset statistics
└── audit/                        # Step 4: Quality control
    ├── audit_report.json         # Comprehensive audit
    └── human_review_samples.json # Human review samples
```

## Dataset Schema

The final dataset follows this schema:

```json
{
  "prompt": "User's input prompt",
  "response": "Model's response",
  "head_a": "REFUSAL.DIRECT",           // Outcome classification
  "head_b": "STYLE.DIRECT",             // Response style (unified)
  "head_c_a": ["weapons", "violence"],  // Harm categories (multilabel)
  "head_c_b": ["technology", "other"],  // Harmless topic categories (multilabel)
  "head_d": {                           // Safety flags (boolean)
    "prompt_harmful": true,
    "response_harmful": false,
    "response_refusal": true
  },
  "metadata": {
    "confidence": 0.85,
    "rationale": "Clear refusal indicators present",
    "judge_model": "gpt-4"
  }
}
```

## Quality Control

### Automated Checks
- **PII Detection**: Automatically detects and flags personal information
- **Toxicity Detection**: Identifies potentially harmful content
- **Deduplication**: Finds and removes duplicate entries
- **Confidence Scoring**: Estimates labeling confidence

### Human Review
- **Stratified Sampling**: Selects samples across different outcomes and confidence levels
- **Review Interface**: Provides structured format for human annotation
- **Agreement Metrics**: Measures inter-judge agreement

### Audit Reports
- **Quality Metrics**: Confidence distributions, label distributions
- **Coverage Analysis**: Ensures balanced representation across categories
- **Recommendations**: Actionable suggestions for improvement

## Troubleshooting

### Common Issues

**1. API Rate Limits**
```bash
# Reduce batch size
uv run dataset-pipeline label --batch-size 5

# Add delays between requests (configure in generation.py)
```

**2. Memory Issues**
```bash
# Process smaller chunks
uv run dataset-pipeline run --datasets wildguard_mix  # One dataset at a time
```

**3. Low Confidence Scores**
```bash
# Review constitutional rules
# Check heuristics in labeling.py
# Consider two-pass evaluation
```

**4. Dataset Access Issues**
```bash
# For gated datasets, authenticate with Hugging Face
huggingface-cli login

# Check dataset availability
uv run dataset-pipeline list
```

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
uv run dataset-pipeline run --config config.json
```

### Validation

```bash
# Validate configuration
uv run dataset-pipeline validate --config config.json

# Check dataset format
uv run dataset-pipeline list
```

## Best Practices

### 1. Dataset Selection
- Start with **WildGuardMix** for safety labels
- Add **SORRY-Bench** for category coverage
- Include **OR-Bench** for over-refusal analysis
- Use **JailbreakBench** for adversarial testing

### 2. Configuration
- Use **two-pass evaluation** for better quality
- Set appropriate **confidence thresholds** (0.7-0.8)
- Enable **PII scrubbing** for production use
- Configure **human sampling** (5-10%)

### 3. Quality Control
- Review **human samples** regularly
- Monitor **agreement metrics**
- Check **coverage** across all dimensions
- Address **recommendations** promptly

### 4. Scaling
- Use **batch processing** for large datasets
- Implement **parallel workers** for efficiency
- Cache **intermediate results**
- Monitor **API rate limits**

## Example Workflows

### Workflow 1: Quick Start (Small Dataset)
```bash
# 1. Ingest a small dataset
uv run dataset-pipeline ingest --datasets sorry_bench --output-dir outputs/quick

# 2. Generate responses
uv run dataset-pipeline generate --input-dir outputs/quick --output-dir outputs/quick/processed

# 3. Label data
uv run dataset-pipeline label --input-file outputs/quick/processed/sorry_bench_data.json

# 4. Audit results
uv run dataset-pipeline audit --input-file outputs/quick/processed/labeling_results.json
```

### Workflow 2: Production Pipeline (Large Dataset)
```bash
# 1. Create configuration
cp examples/dataset_pipeline_config.json my_config.json
# Edit my_config.json with your settings

# 2. Run complete pipeline
uv run dataset-pipeline run --config my_config.json

# 3. Review audit report
cat outputs/dataset_pipeline/audit/audit_report.json

# 4. Human review
# Review outputs/dataset_pipeline/audit/human_review_samples.json
```

### Workflow 3: Custom Dataset
```bash
# 1. Prepare your data in JSON format
# [
#   {"prompt": "question", "response": "answer"},
#   ...
# ]

# 2. Create custom config
# Edit examples/dataset_pipeline_config.json

# 3. Run pipeline
uv run dataset-pipeline run --config my_config.json
```

## Integration with Training

The final dataset is ready for training:

```python
from rejection_detection.data_utils import RejectionDetectionDataset
from torch.utils.data import DataLoader

# Load the processed dataset
dataset = RejectionDetectionDataset("outputs/dataset_pipeline/processed/final_dataset.json")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use with your training loop
for batch in dataloader:
    # batch contains all 5 heads
    # batch["labels"]["head_a"] - outcome labels
    # batch["labels"]["head_b"] - response style labels (unified)
    # batch["labels"]["head_c_a"] - harm category labels
    # batch["labels"]["head_c_b"] - harmless topic category labels
    # batch["labels"]["head_d"] - safety flag labels
    pass
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the audit reports for quality issues
3. Validate your configuration
4. Check the logs for detailed error messages

The pipeline is designed to be robust and provide clear feedback at each step to help you create high-quality training data for your rejection detection model.
