"""Configuration classes for the dataset pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class DatasetSource:
    """Configuration for a dataset source."""
    name: str
    source_type: str  # "huggingface", "github", "local", "arxiv"
    source_path: str
    split: Optional[str] = None
    columns: Optional[Dict[str, str]] = None  # Map dataset columns to standard schema
    filters: Optional[Dict[str, Any]] = None
    license: Optional[str] = None
    description: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    models: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo", "claude-3-haiku"])
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: Optional[str] = None
    jailbreak_wrappers: List[str] = field(default_factory=lambda: [
        "ignore_previous_instructions",
        "roleplay",
        "pretend",
        "hypothetical"
    ])


@dataclass
class LabelingConfig:
    """Configuration for CAI judge labeling."""
    judge_model: str = "gpt-4"
    constitution_file: Optional[str] = None
    confidence_threshold: float = 0.7
    max_retries: int = 3
    batch_size: int = 10
    two_pass_evaluation: bool = True
    use_heuristics: bool = True


@dataclass
class QualityConfig:
    """Configuration for quality control."""
    human_sample_rate: float = 0.05  # 5% human review
    agreement_threshold: float = 0.8
    min_confidence: float = 0.6
    deduplication_threshold: float = 0.9
    pii_scrubbing: bool = True
    toxicity_check: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Dataset sources
    datasets: List[DatasetSource] = field(default_factory=list)
    
    # Processing steps
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Output configuration
    output_dir: Path = Path("outputs/dataset_pipeline")
    raw_dir: Path = Path("raw")
    processed_dir: Path = Path("processed")
    audit_dir: Path = Path("audit")
    
    # Schema configuration
    schema_file: Optional[str] = None
    category_mapping_file: Optional[str] = None
    
    # Processing options
    skip_generation: bool = False
    skip_labeling: bool = False
    skip_quality_control: bool = False
    parallel_workers: int = 4
    random_seed: int = 42


# Default dataset configurations
DEFAULT_DATASETS = [
    DatasetSource(
        name="wildguard_mix",
        source_type="huggingface",
        source_path="allenai/wildguardmix",
        split="wildguardtrain",
        columns={
            "prompt": "prompt",
            "response": "response", 
            "prompt_harmful": "prompt_harm",
            "response_harmful": "response_harm",
            "response_refusal": "response_refusal"
        },
        license="AI2 Responsible Use",
        description="WildGuardMix training data with safety labels"
    ),
    DatasetSource(
        name="wildguard_test",
        source_type="huggingface", 
        source_path="allenai/wildguardmix",
        split="wildguardtest",
        columns={
            "prompt": "prompt",
            "response": "response",
            "prompt_harmful": "prompt_harm", 
            "response_harmful": "response_harm",
            "response_refusal": "response_refusal"
        },
        license="AI2 Responsible Use",
        description="WildGuardMix test data (human annotated, held out for eval)"
    ),
    DatasetSource(
        name="sorry_bench",
        source_type="huggingface",
        source_path="sorry-bench/sorry-bench",
        columns={
            "prompt": "instruction",
            "category": "category"
        },
        license="MIT",
        description="SORRY-Bench: 440 unsafe instructions across 44 categories"
    ),
    DatasetSource(
        name="do_not_answer",
        source_type="github",
        source_path="https://github.com/do-not-answer/dataset",
        columns={
            "prompt": "prompt",
            "harm_type": "harm_type"
        },
        license="MIT",
        description="Do-Not-Answer dataset: prompts that should be refused"
    ),
    DatasetSource(
        name="jailbreak_bench",
        source_type="github", 
        source_path="https://github.com/JailbreakBench/JailbreakBench",
        columns={
            "prompt": "jailbreak_prompt",
            "category": "category",
            "harm_type": "harm_type"
        },
        license="MIT",
        description="JailbreakBench: 100 misuse behaviors in 10 categories"
    ),
    DatasetSource(
        name="or_bench",
        source_type="huggingface",
        source_path="or-bench/or-bench",
        columns={
            "prompt": "prompt",
            "category": "rejection_category"
        },
        license="MIT", 
        description="OR-Bench: 80k over-refusal prompts across 10 categories"
    )
]
