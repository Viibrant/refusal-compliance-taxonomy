# Recommended Open Source Models for Dataset Pipeline

This document lists recommended open source models from Hugging Face that can be used with the dataset pipeline for response generation and labeling.

## Response Generation Models

### Conversational Models (Recommended)

#### Microsoft DialoGPT
- **Models**: `microsoft/DialoGPT-small`, `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- **Size**: 117M, 345M, 774M parameters
- **Use Case**: General conversation, good for refusal/compliance patterns
- **Pros**: Fast, lightweight, good conversation flow
- **Cons**: May not be as sophisticated as larger models

#### Meta LLaMA 2 (Chat)
- **Models**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- **Size**: 7B, 13B parameters
- **Use Case**: High-quality responses, good safety training
- **Pros**: Excellent quality, good safety alignment
- **Cons**: Requires approval, large memory requirements

#### Mistral AI
- **Models**: `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Size**: 7B parameters
- **Use Case**: High-quality instruction following
- **Pros**: Excellent performance, good instruction following
- **Cons**: Large memory requirements

#### Zephyr (Mistral-based)
- **Models**: `HuggingFaceH4/zephyr-7b-alpha`, `HuggingFaceH4/zephyr-7b-beta`
- **Size**: 7B parameters
- **Use Case**: Aligned chat models, good for safety
- **Pros**: Well-aligned, good safety behavior
- **Cons**: Large memory requirements

### Smaller/Faster Models

#### GPT-2 Variants
- **Models**: `gpt2`, `gpt2-medium`, `gpt2-large`
- **Size**: 117M, 345M, 774M parameters
- **Use Case**: Quick generation, baseline comparisons
- **Pros**: Fast, widely available
- **Cons**: Older architecture, less sophisticated

#### DistilGPT-2
- **Models**: `distilgpt2`
- **Size**: 82M parameters
- **Use Case**: Very fast generation, resource-constrained environments
- **Pros**: Very fast, small memory footprint
- **Cons**: Lower quality responses

#### Microsoft CodeGPT
- **Models**: `microsoft/CodeGPT-small-py`, `microsoft/CodeGPT-medium-py`
- **Size**: 117M, 345M parameters
- **Use Case**: Code-related prompts, technical responses
- **Pros**: Good for technical content
- **Cons**: Specialized for code

## CAI Judge Models (for Labeling)

### Recommended for Labeling

#### LLaMA 2 Chat Models
- **Models**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- **Use Case**: High-quality labeling, good reasoning
- **Pros**: Excellent judgment, good safety understanding
- **Cons**: Requires approval, large memory

#### Mistral Instruct
- **Models**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Use Case**: High-quality instruction following for labeling
- **Pros**: Excellent instruction following, good reasoning
- **Cons**: Large memory requirements

#### Zephyr Models
- **Models**: `HuggingFaceH4/zephyr-7b-alpha`
- **Use Case**: Aligned models for consistent labeling
- **Pros**: Well-aligned, consistent behavior
- **Cons**: Large memory requirements

### Fallback Options

#### DialoGPT Large
- **Models**: `microsoft/DialoGPT-large`
- **Use Case**: When larger models aren't available
- **Pros**: Reasonable quality, manageable size
- **Cons**: May not be as sophisticated

## Configuration Examples

### Minimal Setup (CPU-friendly)
```json
{
  "generation": {
    "models": ["microsoft/DialoGPT-medium", "distilgpt2"],
    "temperature": 0.7,
    "max_tokens": 256
  },
  "labeling": {
    "judge_model": "microsoft/DialoGPT-large"
  }
}
```

### Balanced Setup (GPU recommended)
```json
{
  "generation": {
    "models": ["microsoft/DialoGPT-large", "gpt2-large"],
    "temperature": 0.7,
    "max_tokens": 512
  },
  "labeling": {
    "judge_model": "microsoft/DialoGPT-large"
  }
}
```

### High-Quality Setup (GPU required)
```json
{
  "generation": {
    "models": ["mistralai/Mistral-7B-Instruct-v0.1", "HuggingFaceH4/zephyr-7b-alpha"],
    "temperature": 0.7,
    "max_tokens": 512
  },
  "labeling": {
    "judge_model": "mistralai/Mistral-7B-Instruct-v0.1"
  }
}
```

### Production Setup (Multiple GPUs recommended)
```json
{
  "generation": {
    "models": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1"],
    "temperature": 0.7,
    "max_tokens": 512
  },
  "labeling": {
    "judge_model": "meta-llama/Llama-2-13b-chat-hf"
  }
}
```

## Hardware Requirements

### CPU-only Setup
- **RAM**: 8GB minimum, 16GB recommended
- **Models**: DialoGPT-medium, GPT-2 variants, DistilGPT-2
- **Speed**: Slower but functional

### Single GPU Setup
- **GPU**: 8GB VRAM minimum (RTX 3070, RTX 4060 Ti)
- **Models**: DialoGPT-large, GPT-2-large, smaller 7B models
- **Speed**: Good performance

### Multi-GPU Setup
- **GPU**: 16GB+ VRAM per GPU (RTX 4080, RTX 4090, A100)
- **Models**: LLaMA 2, Mistral, Zephyr
- **Speed**: Excellent performance

## Usage Examples

### Command Line Usage

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

### Programmatic Usage

```python
from dataset_pipeline import GenerationConfig, LabelingConfig

# Configure for open source models
gen_config = GenerationConfig(
    models=["microsoft/DialoGPT-large", "mistralai/Mistral-7B-Instruct-v0.1"],
    temperature=0.7,
    max_tokens=512
)

label_config = LabelingConfig(
    judge_model="microsoft/DialoGPT-large",
    confidence_threshold=0.7
)
```

## Model Access Requirements

### No Approval Required
- Microsoft DialoGPT series
- GPT-2 variants
- DistilGPT-2
- Mistral models (some)
- Zephyr models

### Approval Required
- Meta LLaMA 2 models (requires Hugging Face approval)
- Some specialized models

### Setup Instructions

1. **For LLaMA 2 models**:
   ```bash
   # Request access on Hugging Face
   # Then login with your token
   huggingface-cli login
   ```

2. **For other models**:
   ```bash
   # No special setup required
   # Models will be downloaded automatically
   ```

## Performance Tips

### Memory Optimization
- Use smaller models for initial testing
- Enable gradient checkpointing for large models
- Use CPU offloading when possible

### Speed Optimization
- Use GPU when available
- Batch multiple prompts together
- Use smaller max_tokens for faster generation

### Quality Optimization
- Use chat/instruct models for better alignment
- Adjust temperature for desired creativity
- Use two-pass evaluation for better labeling

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Use smaller models
--models microsoft/DialoGPT-medium

# Reduce batch size
--batch-size 1
```

**Model Loading Errors**
```bash
# Check model name spelling
# Ensure you have access to gated models
huggingface-cli login
```

**Slow Generation**
```bash
# Use GPU if available
# Reduce max_tokens
# Use smaller models
```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
uv run dataset-pipeline generate --models microsoft/DialoGPT-medium
```

This setup allows you to use high-quality open source models for your dataset pipeline without relying on commercial APIs!
