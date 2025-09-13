"""Taxonomy definitions for multi-head rejection detection model."""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class HeadConfig:
    """Configuration for a model head."""
    name: str
    labels: List[str]
    num_classes: int
    head_type: str  # "classification", "multilabel", "boolean"


# Head A: Main classification head
HEAD_A_LABELS = [
    "REFUSAL.DIRECT",
    "REFUSAL.PARTIAL", 
    "REFUSAL.CAPABILITY",
    "COMPLY.BENIGN",
    "COMPLY.UNSAFE",
    "COMPLY.TRANSFORM"
]

# Head B.A: Refusal style classification (only for refusal cases)
HEAD_B_A_LABELS = [
    "STYLE.DIRECT",
    "STYLE.EXPLAIN", 
    "STYLE.REDIRECT",
    "STYLE.CLARIFY",
    "STYLE.APPEAL_TO_VALUES",
    "STYLE.RATE_LIMIT"
]

# Head B.B: Compliance style classification (only for compliance cases)
HEAD_B_B_LABELS = [
    "STYLE.DIRECT_ANSWER",
    "STYLE.HIGH_LEVEL",
    "STYLE.STEP_BY_STEP", 
    "STYLE.WITH_WARNINGS",
    "STYLE.REFRACT",
    "STYLE.CLARIFY_THEN_COMPLY",
    "STYLE.TOOL_USE",
    "STYLE.CITED",
    "STYLE.CREATIVE",
    "STYLE.TRANSFORM_ONLY",
    "STYLE.PARTIAL_COMPLY"
]

# Head C: Harm category classification (multilabel)
HEAD_C_LABELS = [
    "weapons",
    "illicit_behavior", 
    "cybercrime",
    "fraud",
    "terrorism",
    "self_harm",
    "hate_harassment",
    "sexual_exploitative",
    "adult_sexual",
    "privacy_personal_data",
    "medical_advice",
    "legal_advice", 
    "financial_advice",
    "extremism",
    "violence",
    "bullying",
    "child_safety",
    "misinformation",
    "controlled_substances",
    "gambling",
    "political_persuasion",
    "biological_threats",
    "chemical_threats",
    "radiological_nuclear",
    "dangerous_activities",
    "other"
]

# Head D: Boolean classification head
HEAD_D_LABELS = [
    "prompt_harmful",
    "response_harmful", 
    "response_refusal"
]


# Head configurations
HEAD_CONFIGS = {
    "head_a": HeadConfig(
        name="head_a",
        labels=HEAD_A_LABELS,
        num_classes=len(HEAD_A_LABELS),
        head_type="classification"
    ),
    "head_b_a": HeadConfig(
        name="head_b_a", 
        labels=HEAD_B_A_LABELS,
        num_classes=len(HEAD_B_A_LABELS),
        head_type="classification"
    ),
    "head_b_b": HeadConfig(
        name="head_b_b",
        labels=HEAD_B_B_LABELS, 
        num_classes=len(HEAD_B_B_LABELS),
        head_type="classification"
    ),
    "head_c": HeadConfig(
        name="head_c",
        labels=HEAD_C_LABELS,
        num_classes=len(HEAD_C_LABELS),
        head_type="multilabel"
    ),
    "head_d": HeadConfig(
        name="head_d",
        labels=HEAD_D_LABELS,
        num_classes=len(HEAD_D_LABELS),
        head_type="boolean"
    )
}


def get_head_config(head_name: str) -> HeadConfig:
    """Get configuration for a specific head."""
    if head_name not in HEAD_CONFIGS:
        raise ValueError(f"Unknown head: {head_name}. Available heads: {list(HEAD_CONFIGS.keys())}")
    return HEAD_CONFIGS[head_name]


def get_all_head_configs() -> Dict[str, HeadConfig]:
    """Get all head configurations."""
    return HEAD_CONFIGS.copy()


def get_label_to_id_mapping(head_name: str) -> Dict[str, int]:
    """Get label to ID mapping for a specific head."""
    config = get_head_config(head_name)
    return {label: idx for idx, label in enumerate(config.labels)}


def get_id_to_label_mapping(head_name: str) -> Dict[int, str]:
    """Get ID to label mapping for a specific head."""
    config = get_head_config(head_name)
    return {idx: label for idx, label in enumerate(config.labels)}


def get_total_num_classes() -> int:
    """Get total number of classes across all heads."""
    return sum(config.num_classes for config in HEAD_CONFIGS.values())


def get_head_names() -> List[str]:
    """Get list of all head names."""
    return list(HEAD_CONFIGS.keys())


def validate_head_dependencies() -> Dict[str, List[str]]:
    """
    Validate head dependencies and return dependency mapping.
    
    Returns:
        Dict mapping head names to their dependencies
    """
    dependencies = {
        "head_a": [],  # No dependencies
        "head_b_a": ["head_a"],  # Depends on head_a being REFUSAL
        "head_b_b": ["head_a"],  # Depends on head_a being COMPLY
        "head_c": [],  # No dependencies
        "head_d": []   # No dependencies
    }
    return dependencies


def get_conditional_heads() -> Dict[str, List[str]]:
    """
    Get heads that are conditionally applied based on other head outputs.
    
    Returns:
        Dict mapping condition to list of heads that should be applied
    """
    return {
        "refusal": ["head_b_a"],  # Apply head_b_a only for refusal cases
        "compliance": ["head_b_b"]  # Apply head_b_b only for compliance cases
    }


def is_refusal_label(label: str) -> bool:
    """Check if a label from head_a indicates refusal."""
    return label.startswith("REFUSAL")


def is_compliance_label(label: str) -> bool:
    """Check if a label from head_a indicates compliance."""
    return label.startswith("COMPLY")


def get_style_head_for_main_label(main_label: str) -> str:
    """
    Get the appropriate style head based on the main classification.
    
    Args:
        main_label: Label from head_a
        
    Returns:
        Name of the style head to use
    """
    if is_refusal_label(main_label):
        return "head_b_a"
    elif is_compliance_label(main_label):
        return "head_b_b"
    else:
        raise ValueError(f"Unknown main label type: {main_label}")


# Utility functions for data processing
def create_label_encoding_dict() -> Dict[str, Dict[str, int]]:
    """Create label encoding dictionaries for all heads."""
    return {
        head_name: get_label_to_id_mapping(head_name)
        for head_name in get_head_names()
    }


def create_reverse_encoding_dict() -> Dict[str, Dict[int, str]]:
    """Create reverse label encoding dictionaries for all heads."""
    return {
        head_name: get_id_to_label_mapping(head_name)
        for head_name in get_head_names()
    }
