# Example Output Files

This directory contains example output files from the rejection detection system.

## Files

### Training Data
- **`train_data.json`** - Training dataset with examples of all taxonomy categories
- **`val_data.json`** - Validation dataset for model evaluation
- **`test_data.json`** - Test dataset for final model assessment

### Training Configuration
- **`training_args.json`** - Training configuration and hyperparameters used

### Inference Results
- **`inference_results.json`** - Example inference results from the trained model

## Data Format

All data files follow the comprehensive taxonomy structure:

### Head A (Outcome) - 8 classes
- `REFUSAL.DIRECT` - Explicit policy/safety refusal
- `REFUSAL.SOFT` - Refusal with empathy/support
- `REFUSAL.PARTIAL` - Refusal plus task content
- `REFUSAL.CAPABILITY` - Refusal due to missing capabilities
- `REFUSAL.OVER` - Unnecessary refusal of safe requests
- `COMPLY.BENIGN` - Helpful answer to safe requests
- `COMPLY.TRANSFORM` - Transform-only operations on risky content
- `COMPLY.UNSAFE` - Provides unsafe/disallowed content

### Head B.A (Refusal Style) - 7 classes
- `STYLE.DIRECT`, `STYLE.EXPLAIN`, `STYLE.REDIRECT`, `STYLE.CLARIFY`, `STYLE.APPEAL_TO_VALUES`, `STYLE.EMPATHETIC`, `STYLE.RATE_LIMIT`

### Head B.B (Compliance Style) - 11 classes
- `STYLE.DIRECT_ANSWER`, `STYLE.HIGH_LEVEL`, `STYLE.STEP_BY_STEP`, `STYLE.WITH_WARNINGS`, `STYLE.REFRACT`, `STYLE.CLARIFY_THEN_COMPLY`, `STYLE.TOOL_USE`, `STYLE.CITED`, `STYLE.CREATIVE`, `STYLE.TRANSFORM_ONLY`, `STYLE.PARTIAL_COMPLY`

### Head C (Harm Categories) - 27 classes (multilabel)
- `weapons`, `illicit_behavior`, `cybercrime`, `fraud`, `terrorism`, `self_harm`, `hate_harassment`, `sexual_exploitative`, `adult_sexual`, `privacy_personal_data`, `medical_advice`, `legal_advice`, `financial_advice`, `extremism`, `violence`, `bullying`, `child_safety`, `misinformation`, `controlled_substances`, `gambling`, `intellectual_property`, `political_persuasion`, `biological_threats`, `chemical_threats`, `radiological_nuclear`, `dangerous_activities`, `other`

### Head D (Binary Flags) - 3 classes
- `prompt_harmful` - Whether the prompt is harmful/unsafe
- `response_harmful` - Whether the response contains unsafe content
- `response_refusal` - Whether the response explicitly refuses

## Usage

These files can be used as:
- **Training examples** for understanding the taxonomy
- **Reference data** for model evaluation
- **Test cases** for validation
- **Documentation** of the expected data format

## Data Processing

Use the data processing CLI to analyze these files:

```bash
# Show statistics
uv run data-processing stats --input examples/outputs/train_data.json

# Validate data format
uv run data-processing validate --input examples/outputs/train_data.json
```
