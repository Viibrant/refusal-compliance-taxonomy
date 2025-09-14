"""Data processing utilities for rejection detection datasets."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from rejection_detection.taxonomies import get_all_head_configs, get_label_to_id_mapping

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    min_prompt_length: int = 10
    max_prompt_length: int = 1000
    min_response_length: int = 5
    max_response_length: int = 2000
    remove_duplicates: bool = True
    balance_classes: bool = False
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


class DataProcessor:
    """Main class for processing rejection detection datasets."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.head_configs = get_all_head_configs()
        
    def load_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            return self._load_json(file_path)
        elif file_path.suffix == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix == '.jsonl':
            return self._load_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("JSON file must contain a list or dict with 'data' key")
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate and filter data based on configuration."""
        valid_data = []
        errors = []
        
        for i, item in enumerate(data):
            try:
                # Check required fields
                if 'prompt' not in item or 'response' not in item:
                    errors.append(f"Item {i}: Missing required fields 'prompt' or 'response'")
                    continue
                
                prompt = str(item['prompt']).strip()
                response = str(item['response']).strip()
                
                # Check length constraints
                if len(prompt) < self.config.min_prompt_length:
                    errors.append(f"Item {i}: Prompt too short ({len(prompt)} < {self.config.min_prompt_length})")
                    continue
                
                if len(prompt) > self.config.max_prompt_length:
                    errors.append(f"Item {i}: Prompt too long ({len(prompt)} > {self.config.max_prompt_length})")
                    continue
                
                if len(response) < self.config.min_response_length:
                    errors.append(f"Item {i}: Response too short ({len(response)} < {self.config.min_response_length})")
                    continue
                
                if len(response) > self.config.max_response_length:
                    errors.append(f"Item {i}: Response too long ({len(response)} > {self.config.max_response_length})")
                    continue
                
                # Validate labels if present
                item_errors = self._validate_labels(item, i)
                if item_errors:
                    errors.extend(item_errors)
                    continue
                
                valid_data.append(item)
                
            except Exception as e:
                errors.append(f"Item {i}: Validation error - {str(e)}")
        
        logger.info(f"Validation complete: {len(valid_data)} valid items, {len(errors)} errors")
        return valid_data, errors
    
    def _validate_labels(self, item: Dict[str, Any], index: int) -> List[str]:
        """Validate labels for all heads."""
        errors = []
        
        for head_name, head_config in self.head_configs.items():
            if head_name not in item:
                continue  # Optional labels
            
            label = item[head_name]
            
            if head_config.head_type == "multilabel":
                if not isinstance(label, list):
                    errors.append(f"Item {index}: {head_name} should be a list for multilabel")
                    continue
                
                # Check if all labels are valid
                label_to_id = get_label_to_id_mapping(head_name)
                for l in label:
                    if l not in label_to_id:
                        errors.append(f"Item {index}: Invalid label '{l}' for {head_name}")
            elif head_config.head_type == "boolean":
                # Boolean head - should be a dict with boolean values
                if not isinstance(label, dict):
                    errors.append(f"Item {index}: {head_name} should be a dict for boolean head")
                    continue
                
                # Check if all values are boolean
                for key, value in label.items():
                    if not isinstance(value, bool):
                        errors.append(f"Item {index}: {head_name}.{key} should be a boolean")
            else:
                # Single label
                if not isinstance(label, str):
                    errors.append(f"Item {index}: {head_name} should be a string for single label")
                    continue
                
                label_to_id = get_label_to_id_mapping(head_name)
                if label not in label_to_id:
                    errors.append(f"Item {index}: Invalid label '{label}' for {head_name}")
        
        return errors
    
    def remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entries based on prompt-response pairs."""
        if not self.config.remove_duplicates:
            return data
        
        seen = set()
        unique_data = []
        
        for item in data:
            key = (item['prompt'].strip().lower(), item['response'].strip().lower())
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        removed = len(data) - len(unique_data)
        logger.info(f"Removed {removed} duplicate entries")
        return unique_data
    
    def balance_classes(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance classes by undersampling majority classes."""
        if not self.config.balance_classes:
            return data
        
        # For now, just return the data as-is
        # TODO: Implement proper class balancing
        logger.info("Class balancing not yet implemented")
        return data
    
    def split_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train, validation, and test sets."""
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(len(data))
        
        n_test = int(len(data) * self.config.test_split)
        n_val = int(len(data) * self.config.validation_split)
        n_train = len(data) - n_test - n_val
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def process_dataset(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Complete dataset processing pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing dataset from {input_path}")
        
        # Load data
        data = self.load_data(input_path)
        logger.info(f"Loaded {len(data)} items")
        
        # Validate data
        valid_data, errors = self.validate_data(data)
        
        # Save validation errors
        if errors:
            error_file = output_dir / "validation_errors.txt"
            with open(error_file, 'w') as f:
                f.write('\n'.join(errors))
            logger.warning(f"Saved {len(errors)} validation errors to {error_file}")
        
        # Remove duplicates
        unique_data = self.remove_duplicates(valid_data)
        
        # Balance classes
        balanced_data = self.balance_classes(unique_data)
        
        # Split data
        train_data, val_data, test_data = self.split_data(balanced_data)
        
        # Save processed data
        self._save_data(train_data, output_dir / "train.json")
        self._save_data(val_data, output_dir / "val.json")
        self._save_data(test_data, output_dir / "test.json")
        
        # Save processing report
        report = {
            "input_file": str(input_path),
            "total_items": len(data),
            "valid_items": len(valid_data),
            "unique_items": len(unique_data),
            "final_items": len(balanced_data),
            "train_items": len(train_data),
            "val_items": len(val_data),
            "test_items": len(test_data),
            "validation_errors": len(errors),
            "config": {
                "min_prompt_length": self.config.min_prompt_length,
                "max_prompt_length": self.config.max_prompt_length,
                "min_response_length": self.config.min_response_length,
                "max_response_length": self.config.max_response_length,
                "remove_duplicates": self.config.remove_duplicates,
                "balance_classes": self.config.balance_classes,
                "validation_split": self.config.validation_split,
                "test_split": self.config.test_split,
                "random_seed": self.config.random_seed
            }
        }
        
        self._save_data(report, output_dir / "processing_report.json")
        
        logger.info(f"Processing complete. Output saved to {output_dir}")
        return report
    
    def _save_data(self, data: Union[List[Dict], Dict], file_path: Path):
        """Save data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
