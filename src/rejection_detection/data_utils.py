"""Data utilities for multi-head rejection detection."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, load_dataset
import pandas as pd
import numpy as np

from .taxonomies import (
    get_head_configs,
    get_head_config,
    get_label_to_id_mapping,
    is_refusal_label,
    is_compliance_label,
    OutcomeType,
    ResponseStyle,
    HarmCategory,
)

logger = logging.getLogger(__name__)


class RejectionDetectionDataset(Dataset):
    """Dataset for rejection detection with multi-head labels."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of data samples
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mappings
        self.label_mappings = {
            head_name: get_label_to_id_mapping(head_name)
            for head_name in get_head_configs().keys()
        }
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate the dataset format."""
        if not self.data:
            raise ValueError("Dataset is empty")
        
        # Check required fields
        required_fields = ["prompt", "response"]
        for i, sample in enumerate(self.data[:5]):  # Check first 5 samples
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Sample {i} missing required field: {field}")
        
        # Check label fields
        expected_label_fields = ["head_a", "head_b", "head_c_a", "head_c_b", "head_d"]
        
        for i, sample in enumerate(self.data[:5]):
            for field in expected_label_fields:
                if field not in sample:
                    logger.warning(f"Sample {i} missing label field: {field}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Prepare text
        prompt = sample["prompt"]
        response = sample["response"]
        
        # Combine prompt and response
        text = f"{prompt} [SEP] {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Prepare labels
        labels = {}
        
        # Head A: Main classification
        if "head_a" in sample:
            head_a_label = sample["head_a"]
            if isinstance(head_a_label, str):
                labels["head_a"] = torch.tensor(
                    self.label_mappings["head_a"][head_a_label],
                    dtype=torch.long
                )
            else:
                labels["head_a"] = torch.tensor(head_a_label, dtype=torch.long)
        
        # Head B: Response style
        if "head_b" in sample:
            head_b_label = sample["head_b"]
            if isinstance(head_b_label, str):
                labels["head_b"] = torch.tensor(
                    self.label_mappings["head_b"][head_b_label],
                    dtype=torch.long
                )
            else:
                labels["head_b"] = torch.tensor(head_b_label, dtype=torch.long)
        
        
        # Head C.A: Harm categories (multilabel)
        if "head_c_a" in sample:
            head_c_a_labels = sample["head_c_a"]
            if isinstance(head_c_a_labels, list):
                # Convert list of labels to binary vector
                head_configs = get_head_configs()
                label_vector = torch.zeros(len(head_configs["head_c_a"].class_names), dtype=torch.float)
                for label in head_c_a_labels:
                    if label in self.label_mappings["head_c_a"]:
                        label_vector[self.label_mappings["head_c_a"][label]] = 1.0
                labels["head_c_a"] = label_vector
            else:
                # Assume it's already a binary vector
                labels["head_c_a"] = torch.tensor(head_c_a_labels, dtype=torch.float)
        
        # Head C.B: Harmless topic categories (multilabel)
        if "head_c_b" in sample:
            head_c_b_labels = sample["head_c_b"]
            if isinstance(head_c_b_labels, list):
                # Convert list of labels to binary vector
                head_configs = get_head_configs()
                label_vector = torch.zeros(len(head_configs["head_c_b"].class_names), dtype=torch.float)
                for label in head_c_b_labels:
                    if label in self.label_mappings["head_c_b"]:
                        label_vector[self.label_mappings["head_c_b"][label]] = 1.0
                labels["head_c_b"] = label_vector
            else:
                # Assume it's already a binary vector
                labels["head_c_b"] = torch.tensor(head_c_b_labels, dtype=torch.float)
        
        # Head D: Boolean labels
        if "head_d" in sample:
            head_d_labels = sample["head_d"]
            if isinstance(head_d_labels, dict):
                # Convert dict to binary vector
                head_configs = get_head_configs()
                label_vector = torch.zeros(len(head_configs["head_d"].class_names), dtype=torch.float)
                for i, label_name in enumerate(head_configs["head_d"].class_names):
                    if label_name in head_d_labels:
                        label_vector[i] = float(head_d_labels[label_name])
                labels["head_d"] = label_vector
            else:
                # Assume it's already a binary vector
                labels["head_d"] = torch.tensor(head_d_labels, dtype=torch.float)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": labels,
        }


def load_data_from_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # If it's a dict, assume it has a "data" key
        data = data.get("data", data)
    
    if not isinstance(data, list):
        raise ValueError("Data should be a list of samples")
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def load_data_from_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a CSV file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert DataFrame to list of dicts
    data = df.to_dict("records")
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def load_data_from_huggingface(
    dataset_name: str,
    split: str = "train",
    **kwargs
) -> List[Dict[str, Any]]:
    """Load data from Hugging Face datasets."""
    try:
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        data = dataset.to_list()
        logger.info(f"Loaded {len(data)} samples from Hugging Face dataset: {dataset_name}")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def create_dataloader(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    dataset = RejectionDetectionDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    def collate_fn(batch):
        """Custom collate function for batching."""
        # Separate inputs and labels
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        
        # Collect all unique label keys
        all_label_keys = set()
        for item in batch:
            all_label_keys.update(item["labels"].keys())
        
        # Create labels dict
        labels = {}
        for key in all_label_keys:
            labels[key] = torch.stack([
                item["labels"].get(key, torch.tensor(0))
                for item in batch
            ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data)
    
    # Calculate split indices
    n_samples = len(shuffled_data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def create_sample_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create sample data for testing."""
    sample_data = []
    
    # Sample labels
    head_a_labels = [
        "REFUSAL.DIRECT", "REFUSAL.SOFT", "REFUSAL.PARTIAL", "REFUSAL.CAPABILITY", "REFUSAL.OVER",
        "COMPLY.BENIGN", "COMPLY.UNSAFE", "COMPLY.TRANSFORM", "COMPLY.CONDITIONAL",
        "COMPLY.EDUCATIONAL", "COMPLY.REDIRECTED", "COMPLY.PARTIAL_SAFE"
    ]
    
    head_b_labels = [
        "STYLE.DIRECT", "STYLE.EXPLAIN", "STYLE.EMPATHETIC", "STYLE.CLARIFY_TO_PROCEED",
        "STYLE.APPEAL_TO_VALUES", "STYLE.DIRECT_ANSWER", "STYLE.HIGH_LEVEL", "STYLE.STEP_BY_STEP",
        "STYLE.WITH_WARNINGS", "STYLE.REFRACT", "STYLE.CLARIFY_THEN_COMPLY", "STYLE.TOOL_USE",
        "STYLE.CITED", "STYLE.CREATIVE", "STYLE.TRANSFORM_ONLY", "STYLE.PARTIAL_COMPLY",
        "STYLE.ANALOGY", "STYLE.SPECULATIVE", "STYLE.ELABORATIVE", "STYLE.INTERACTIVE",
        "STYLE.PERSONA_BASED", "STYLE.HUMOROUS", "STYLE.FORMAL"
    ]
    
    head_c_a_labels = [
        "weapons", "illicit_behavior", "cybercrime", "fraud", "terrorism",
        "self_harm", "hate_harassment", "sexual_exploitative", "adult_sexual",
        "privacy_personal_data", "medical_advice", "legal_advice",
        "financial_advice", "extremism", "violence", "bullying",
        "child_safety", "misinformation", "controlled_substances",
        "gambling", "political_persuasion", "biological_threats",
        "chemical_threats", "radiological_nuclear", "dangerous_activities", "other"
    ]
    
    head_c_b_labels = [
        "health_medicine", "education", "technology", "science_space",
        "environment_climate", "politics_government", "law_legal",
        "business_entrepreneurship", "finance_economy", "sports",
        "travel_tourism", "food_cooking", "entertainment", "arts_literature",
        "history", "psychology_self_development", "religion_philosophy",
        "lifestyle_home", "relationships_dating", "other"
    ]
    
    for i in range(num_samples):
        # Random main classification
        head_a = np.random.choice(head_a_labels)
        
        # Random response style (unified head_b)
        head_b = np.random.choice(head_b_labels)
        
        # Random harm categories (multilabel)
        num_harm_categories = np.random.randint(0, 4)
        head_c_a = np.random.choice(head_c_a_labels, size=num_harm_categories, replace=False).tolist()
        
        # Random harmless topic categories (multilabel)
        num_harmless_categories = np.random.randint(0, 3)
        head_c_b = np.random.choice(head_c_b_labels, size=num_harmless_categories, replace=False).tolist()
        
        # Random boolean labels
        head_d = {
            "prompt_harmful": np.random.choice([True, False]),
            "response_harmful": np.random.choice([True, False]),
            "response_refusal": head_a.startswith("REFUSAL")
        }
        
        sample = {
            "prompt": f"Sample prompt {i}: How do I make a bomb?",
            "response": f"Sample response {i}: I cannot help with that request.",
            "head_a": head_a,
            "head_b": head_b,
            "head_c_a": head_c_a,
            "head_c_b": head_c_b,
            "head_d": head_d,
        }
        
        sample_data.append(sample)
    
    return sample_data


def save_data(data: List[Dict[str, Any]], file_path: Union[str, Path]):
    """Save data to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    converted_data = convert_numpy_types(data)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} samples to {file_path}")
