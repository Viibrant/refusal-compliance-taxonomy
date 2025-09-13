# Dataset Pipeline - Quick Reference

## Essential Commands

### List Available Datasets
```bash
uv run dataset-pipeline list
```

### Run Complete Pipeline
```bash
# Basic usage
uv run dataset-pipeline run --output-dir outputs/my_dataset

# With specific datasets
uv run dataset-pipeline run --datasets wildguard_mix sorry_bench --output-dir outputs/custom

# With custom config
uv run dataset-pipeline run --config examples/dataset_pipeline_config.json
```

### Individual Steps

#### 1. Ingest Datasets
```bash
uv run dataset-pipeline ingest --datasets wildguard_mix sorry_bench --output-dir outputs/raw
```

#### 2. Generate Responses
```bash
uv run dataset-pipeline generate --input-dir outputs/raw --output-dir outputs/generated
```

#### 3. Label Data
```bash
uv run dataset-pipeline label --input-file data.json --output-file labeled.json --judge-model gpt-4
```

#### 4. Audit Quality
```bash
uv run dataset-pipeline audit --input-file labeled.json --output-dir outputs/audit
```

## Common Options

### Pipeline Options
- `--output-dir` - Output directory
- `--datasets` - Specific datasets to process
- `--skip-generation` - Skip response generation
- `--skip-labeling` - Skip CAI judge labeling
- `--skip-quality` - Skip quality control
- `--parallel-workers` - Number of parallel workers
- `--random-seed` - Random seed for reproducibility

### Generation Options
- `--models` - Models to use (gpt-3.5-turbo, claude-3-haiku, etc.)
- `--temperature` - Generation temperature (0.0-1.0)
- `--max-tokens` - Maximum tokens to generate

### Labeling Options
- `--judge-model` - Judge model (gpt-4, claude-3-opus, etc.)
- `--confidence-threshold` - Confidence threshold (0.0-1.0)
- `--batch-size` - Batch size for labeling

### Quality Options
- `--human-sample-rate` - Human review sample rate (0.0-1.0)
- `--min-confidence` - Minimum confidence threshold

## Available Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| `wildguard_mix` | Hugging Face | 86K+ | WildGuardMix training data with safety labels |
| `wildguard_test` | Hugging Face | 1.7K | WildGuardMix test data (human annotated) |
| `sorry_bench` | Hugging Face | 440 | SORRY-Bench unsafe instructions (44 categories) |
| `do_not_answer` | GitHub | Variable | Prompts that should be refused |
| `jailbreak_bench` | GitHub | 100 | JailbreakBench misuse behaviors (10 categories) |
| `or_bench` | Hugging Face | 80K | OR-Bench over-refusal prompts (10 categories) |

## Output Files

### Raw Data (`raw/`)
- `{dataset}_data.json` - Ingested data
- `{dataset}_metadata.json` - Dataset metadata
- `{dataset}_provenance.json` - Data provenance

### Processed Data (`processed/`)
- `{dataset}_generated_responses.json` - Generated responses
- `labeling_results.json` - CAI judge labeling results
- `final_dataset.json` - Final multi-head labeled dataset
- `final_dataset.jsonl` - Line-delimited format
- `dataset_info.json` - Dataset statistics

### Audit Data (`audit/`)
- `audit_report.json` - Comprehensive quality audit
- `human_review_samples.json` - Human review samples

## Multi-Head Schema

### Head A: Outcome
- `REFUSAL.DIRECT` - Direct refusal
- `REFUSAL.PARTIAL` - Partial refusal with explanation
- `REFUSAL.CAPABILITY` - Capability-based refusal
- `COMPLY.BENIGN` - Benign compliance
- `COMPLY.UNSAFE` - Unsafe compliance
- `COMPLY.TRANSFORM` - Transformed compliance

### Head B.A: Refusal Style
- `STYLE.DIRECT` - Direct, clear refusal
- `STYLE.EXPLAIN` - Refusal with explanation
- `STYLE.REDIRECT` - Refusal with redirection
- `STYLE.CLARIFY` - Refusal with clarification request
- `STYLE.APPEAL_TO_VALUES` - Refusal appealing to values
- `STYLE.RATE_LIMIT` - Refusal citing rate limits

### Head B.B: Compliance Style
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
`weapons`, `violence`, `self_harm`, `hate_harassment`, `cybercrime`, `fraud`, `terrorism`, `sexual_exploitative`, `privacy_personal_data`, `medical_advice`, `legal_advice`, `financial_advice`, `extremism`, `bullying`, `child_safety`, `misinformation`, `controlled_substances`, `gambling`, `political_persuasion`, `biological_threats`, `chemical_threats`, `radiological_nuclear`, `dangerous_activities`, `other`

### Head D: Safety Flags (boolean)
- `prompt_harmful` - Whether the prompt is harmful
- `response_harmful` - Whether the response is harmful
- `response_refusal` - Whether the response is a refusal

## Environment Variables

```bash
# API Keys (required for generation and labeling)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Hugging Face (for gated datasets)
export HUGGINGFACE_TOKEN="your-hf-token"

# Logging
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

## Troubleshooting

### Common Issues

**Dataset Access Error**
```bash
# Authenticate with Hugging Face
huggingface-cli login
```

**API Rate Limits**
```bash
# Reduce batch size
uv run dataset-pipeline label --batch-size 5
```

**Memory Issues**
```bash
# Process one dataset at a time
uv run dataset-pipeline run --datasets wildguard_mix
```

**Low Confidence**
```bash
# Check constitutional rules
# Review heuristics in labeling.py
# Enable two-pass evaluation
```

### Debug Commands

```bash
# Validate configuration
uv run dataset-pipeline validate --config config.json

# Check dataset availability
uv run dataset-pipeline list

# Enable debug logging
export LOG_LEVEL=DEBUG
uv run dataset-pipeline run --config config.json
```

## Example Workflows

### Quick Test
```bash
uv run dataset-pipeline ingest --datasets sorry_bench --output-dir outputs/test
uv run dataset-pipeline generate --input-dir outputs/test --output-dir outputs/test/processed
uv run dataset-pipeline label --input-file outputs/test/processed/sorry_bench_data.json
```

### Production Run
```bash
uv run dataset-pipeline run --config examples/dataset_pipeline_config.json
```

### Custom Dataset
```bash
# 1. Prepare data.json with {"prompt": "...", "response": "..."} format
# 2. Edit config.json to point to your data
# 3. Run pipeline
uv run dataset-pipeline run --config my_config.json
```

## Integration

### Load Final Dataset for Training
```python
from rejection_detection.data_utils import RejectionDetectionDataset

dataset = RejectionDetectionDataset("outputs/dataset_pipeline/processed/final_dataset.json")
```

### Programmatic Usage
```python
from dataset_pipeline import DatasetPipeline, PipelineConfig

config = PipelineConfig(output_dir="outputs/my_dataset")
pipeline = DatasetPipeline(config)
result = pipeline.run_pipeline()
```
