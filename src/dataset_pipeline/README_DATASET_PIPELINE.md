# Dataset Processing Pipeline

A comprehensive pipeline for processing and labeling datasets for rejection detection model training. This pipeline implements the end-to-end workflow described in the research literature for creating high-quality, multi-axis labeled datasets.

## Overview

The dataset pipeline processes multiple sources of data to create a unified, multi-head labeled dataset suitable for training rejection detection models. It implements:

- **Dataset Ingestion**: Load data from Hugging Face, GitHub, local files, and arXiv
- **Response Generation**: Generate responses for prompt-only datasets using multiple LLMs
- **CAI Judge Labeling**: Use Constitutional AI principles for comprehensive labeling
- **Quality Control**: Audit, validate, and ensure data quality
- **Multi-Head Schema**: Support for 5 classification heads (A, B.A, B.B, C, D)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   Ingestion      │───▶│   Generation    │───▶│   CAI Judge     │
│                 │    │                  │    │                 │    │                 │
│ • WildGuardMix  │    │ • Hugging Face   │    │ • Multi-LLM     │    │ • Constitutional│
│ • SORRY-Bench   │    │ • GitHub         │    │ • Jailbreak     │    │ • Two-pass      │
│ • Do-Not-Answer │    │ • Local files    │    │ • Temperature   │    │ • Heuristics    │
│ • JailbreakBench│    │ • Column mapping │    │ • System prompts│    │ • Confidence    │
│ • OR-Bench      │    │ • Filtering      │    │ • Batch process │    │ • Rationale     │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐           │
│   Final Dataset │◀───│   Quality Control│◀───│   Data Prep     │◀──────────┘
│                 │    │                  │    │                 │
│ • Multi-head    │    │ • PII Detection  │    │ • Deduplication │
│ • JSON/JSONL    │    │ • Toxicity Check │    │ • Validation    │
│ • Provenance    │    │ • Human Sampling │    │ • Splitting     │
│ • Metadata      │    │ • Agreement      │    │ • Balancing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

```bash
# Install the package with all dependencies
uv sync

# The dataset pipeline CLI will be available as 'dataset-pipeline'
```

## Quick Start

### 1. List Available Datasets

```bash
uv run dataset-pipeline list
```

### 2. Run Complete Pipeline

```bash
# Run with default configuration
uv run dataset-pipeline run --output-dir outputs/my_dataset

# Run with custom configuration
uv run dataset-pipeline run --config examples/dataset_pipeline_config.json

# Run specific datasets only
uv run dataset-pipeline run --datasets wildguard_mix sorry_bench
```

### 3. Run Individual Steps

```bash
# Ingest datasets only
uv run dataset-pipeline ingest --datasets wildguard_mix

# Generate responses only
uv run dataset-pipeline generate --input-dir outputs/dataset_pipeline/raw

# Label data only
uv run dataset-pipeline label --input-file data.json --judge-model gpt-4

# Audit labeled data only
uv run dataset-pipeline audit --input-file labeled_data.json
```

## Configuration

### Pipeline Configuration

Create a configuration file (see `examples/dataset_pipeline_config.json`):

```json
{
  "datasets": [
    {
      "name": "wildguard_mix",
      "source_type": "huggingface",
      "source_path": "allenai/wildguardmix",
      "split": "wildguardtrain",
      "columns": {
        "prompt": "prompt",
        "response": "response",
        "prompt_harmful": "prompt_harm",
        "response_harmful": "response_harm",
        "response_refusal": "response_refusal"
      }
    }
  ],
  "generation": {
    "models": ["gpt-3.5-turbo", "claude-3-haiku"],
    "temperature": 0.7,
    "max_tokens": 512
  },
  "labeling": {
    "judge_model": "gpt-4",
    "confidence_threshold": 0.7,
    "two_pass_evaluation": true
  },
  "quality": {
    "human_sample_rate": 0.05,
    "min_confidence": 0.6,
    "pii_scrubbing": true
  }
}
```

### Constitutional Rules

Define labeling rules in a constitution file (see `examples/dataset_pipeline_constitution.json`):

```json
{
  "harm_categories": ["weapons", "violence", "self_harm", ...],
  "refusal_indicators": ["I can't", "I won't", "not allowed", ...],
  "compliance_indicators": ["Here's how", "I can help you", ...],
  "labeling_rules": {
    "outcome_determination": {
      "REFUSAL.DIRECT": "prompt_harmful=true AND response_refusal=true",
      "COMPLY.BENIGN": "prompt_harmful=false AND response_refusal=false"
    }
  }
}
```

## Supported Datasets

### WildGuardMix (AllenAI)
- **Source**: Hugging Face (`allenai/wildguardmix`)
- **Size**: ~86,759 items (48,783 prompt-only; 37,976 prompt+response)
- **Labels**: prompt_harm, response_harm, response_refusal
- **Use**: Primary source for safety flags (Head D) and outcome bootstrapping

### SORRY-Bench
- **Source**: Hugging Face (`sorry-bench/sorry-bench`)
- **Size**: 440 unsafe instructions across 44 categories
- **Labels**: Category classification
- **Use**: Balanced topic coverage and robustness testing

### Do-Not-Answer (DNA)
- **Source**: GitHub repository
- **Size**: Variable (prompts that should be refused)
- **Labels**: Harm type classification
- **Use**: Refusal pattern analysis

### JailbreakBench (JBB)
- **Source**: GitHub repository
- **Size**: 100 misuse behaviors in 10 categories
- **Labels**: Category and harm type
- **Use**: Adversarial testing and jailbreak analysis

### OR-Bench (Over-Refusal)
- **Source**: Hugging Face (`or-bench/or-bench`)
- **Size**: 80,000 over-refusal prompts
- **Labels**: Rejection category
- **Use**: Over-refusal pattern identification

## Multi-Head Schema

The pipeline creates labeled data with 5 classification heads:

### Head A: Outcome Classification
- `REFUSAL.DIRECT` - Direct refusal
- `REFUSAL.PARTIAL` - Partial refusal with explanation
- `REFUSAL.CAPABILITY` - Capability-based refusal
- `COMPLY.BENIGN` - Benign compliance
- `COMPLY.UNSAFE` - Unsafe compliance
- `COMPLY.TRANSFORM` - Transformed compliance

### Head B.A: Refusal Style (for refusals only)
- `STYLE.DIRECT` - Direct, clear refusal
- `STYLE.EXPLAIN` - Refusal with explanation
- `STYLE.REDIRECT` - Refusal with redirection
- `STYLE.CLARIFY` - Refusal with clarification request
- `STYLE.APPEAL_TO_VALUES` - Refusal appealing to values
- `STYLE.RATE_LIMIT` - Refusal citing rate limits

### Head B.B: Compliance Style (for compliance only)
- `STYLE.DIRECT_ANSWER` - Direct answer
- `STYLE.HIGH_LEVEL` - High-level overview
- `STYLE.STEP_BY_STEP` - Detailed instructions
- `STYLE.WITH_WARNINGS` - Answer with warnings
- `STYLE.REFRACT` - Reframed request
- `STYLE.CLARIFY_THEN_COMPLY` - Clarification then compliance
- `STYLE.TOOL_USE` - Using external tools
- `STYLE.CITED` - Answer with citations
- `STYLE.CREATIVE` - Creative interpretation
- `STYLE.TRANSFORM_ONLY` - Only transformed approach
- `STYLE.PARTIAL_COMPLY` - Partial compliance

### Head C: Harm Categories (multilabel)
- `weapons`, `violence`, `self_harm`, `hate_harassment`, `cybercrime`, `fraud`, `terrorism`, `sexual_exploitative`, `privacy_personal_data`, `medical_advice`, `legal_advice`, `financial_advice`, `extremism`, `bullying`, `child_safety`, `misinformation`, `controlled_substances`, `gambling`, `political_persuasion`, `biological_threats`, `chemical_threats`, `radiological_nuclear`, `dangerous_activities`, `other`

### Head D: Safety Flags (boolean)
- `prompt_harmful` - Whether the prompt is harmful
- `response_harmful` - Whether the response is harmful
- `response_refusal` - Whether the response is a refusal

## Quality Control

The pipeline includes comprehensive quality control:

### Automated Checks
- **PII Detection**: Email, phone, SSN, credit card, address patterns
- **Toxicity Detection**: Harmful keyword identification
- **Deduplication**: Similarity-based duplicate detection
- **Confidence Scoring**: Heuristic-based confidence estimation

### Human Review
- **Stratified Sampling**: 5% human review by confidence and outcome
- **Agreement Metrics**: Inter-judge agreement measurement
- **Coverage Analysis**: Distribution analysis across all dimensions

### Audit Reports
- **Quality Metrics**: Confidence distributions, label distributions
- **Coverage Analysis**: Outcome, category, and style coverage
- **Recommendations**: Actionable improvement suggestions

## Output Structure

```
outputs/dataset_pipeline/
├── raw/                          # Raw ingested data
│   ├── wildguard_mix_data.json
│   ├── wildguard_mix_metadata.json
│   └── wildguard_mix_provenance.json
├── processed/                    # Processed data
│   ├── wildguard_mix_generated_responses.json
│   ├── labeling_results.json
│   ├── final_dataset.json
│   ├── final_dataset.jsonl
│   └── dataset_info.json
└── audit/                        # Quality control reports
    ├── audit_report.json
    └── human_review_samples.json
```

## API Usage

### Programmatic Usage

```python
from dataset_pipeline import DatasetPipeline, PipelineConfig
from dataset_pipeline.config import DEFAULT_DATASETS

# Create configuration
config = PipelineConfig(
    output_dir=Path("outputs/my_dataset"),
    datasets=DEFAULT_DATASETS[:2]  # Use first 2 datasets
)

# Run pipeline
pipeline = DatasetPipeline(config)
result = pipeline.run_pipeline()

if result.success:
    print(f"Processed {result.processed_items} items")
    print(f"Output: {result.output_dir}")
```

### Individual Components

```python
from dataset_pipeline import DatasetIngester, ResponseGenerator, CAIJudge

# Ingest data
ingester = DatasetIngester()
dataset = ingester.ingest_dataset(DEFAULT_DATASETS[0])

# Generate responses
generator = ResponseGenerator(GenerationConfig())
responses = generator.generate_responses(prompts)

# Label data
judge = CAIJudge(LabelingConfig())
results = judge.label_prompts_responses(data)
```

## Best Practices

### 1. Dataset Selection
- Start with WildGuardMix for safety labels
- Add SORRY-Bench for category coverage
- Include OR-Bench for over-refusal analysis
- Use JailbreakBench for adversarial testing

### 2. Configuration
- Use two-pass evaluation for better quality
- Set appropriate confidence thresholds
- Enable PII scrubbing for production use
- Configure human sampling for quality control

### 3. Quality Control
- Review human samples regularly
- Monitor agreement metrics
- Check coverage across all dimensions
- Address recommendations promptly

### 4. Scaling
- Use batch processing for large datasets
- Implement parallel workers for efficiency
- Cache intermediate results
- Monitor API rate limits

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce batch size or add delays
2. **Memory Issues**: Process datasets in smaller chunks
3. **Low Confidence**: Review constitution rules and heuristics
4. **Poor Coverage**: Add more diverse datasets

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uv run dataset-pipeline run --config config.json
```

### Validation

```bash
# Validate configuration
uv run dataset-pipeline validate --config config.json

# Check dataset availability
uv run dataset-pipeline list
```

## Contributing

1. Add new dataset sources in `config.py`
2. Extend constitutional rules in `labeling.py`
3. Improve heuristics in `quality.py`
4. Add new CLI commands in `cli.py`

## License

This pipeline is part of the rejection-detection project and follows the same license terms.
