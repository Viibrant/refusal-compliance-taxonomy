"""Dataset ingestion module for loading data from various sources."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import requests
from datasets import load_dataset, Dataset
import numpy as np

from .config import DatasetSource

logger = logging.getLogger(__name__)


@dataclass
class IngestedDataset:
    """Container for ingested dataset with metadata."""
    name: str
    data: List[Dict[str, Any]]
    source: DatasetSource
    metadata: Dict[str, Any]
    provenance: Dict[str, Any]


class DatasetIngester:
    """Handles ingestion of datasets from various sources."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("cache/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_dataset(self, source: DatasetSource) -> IngestedDataset:
        """Ingest a dataset from the specified source."""
        logger.info(f"Ingesting dataset: {source.name} from {source.source_type}")
        
        if source.source_type == "huggingface":
            return self._ingest_huggingface(source)
        elif source.source_type == "github":
            return self._ingest_github(source)
        elif source.source_type == "local":
            return self._ingest_local(source)
        elif source.source_type == "arxiv":
            return self._ingest_arxiv(source)
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
    
    def _ingest_huggingface(self, source: DatasetSource) -> IngestedDataset:
        """Ingest dataset from Hugging Face Hub."""
        try:
            # Load dataset
            if source.split:
                # For datasets with configurations, use the split as config name
                dataset = load_dataset(source.source_path, source.split)
            else:
                dataset = load_dataset(source.source_path)
                # Use first split if no specific split specified
                dataset = dataset[list(dataset.keys())[0]]
            
            # Handle DatasetDict case (when split is specified but returns a dict)
            if hasattr(dataset, 'keys') and not isinstance(dataset, Dataset):
                # If it's a DatasetDict, use the first available split
                available_splits = list(dataset.keys())
                if available_splits:
                    dataset = dataset[available_splits[0]]
                    logger.info(f"Using split '{available_splits[0]}' from DatasetDict")
                else:
                    raise ValueError("No splits available in DatasetDict")
            
            # Convert to list of dicts
            if isinstance(dataset, Dataset):
                data = dataset.to_list()
            else:
                data = list(dataset)
            
            # Apply column mapping if specified
            if source.columns:
                data = self._apply_column_mapping(data, source.columns)
            
            # Apply filters if specified
            if source.filters:
                data = self._apply_filters(data, source.filters)
            
            # Create metadata
            metadata = {
                "total_items": len(data),
                "columns": list(data[0].keys()) if data else [],
                "source_type": "huggingface",
                "source_path": source.source_path,
                "split": source.split
            }
            
            # Create provenance
            provenance = {
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "source": source.source_path,
                "license": source.license,
                "description": source.description
            }
            
            return IngestedDataset(
                name=source.name,
                data=data,
                source=source,
                metadata=metadata,
                provenance=provenance
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest Hugging Face dataset {source.name}: {e}")
            raise
    
    def _ingest_github(self, source: DatasetSource) -> IngestedDataset:
        """Ingest dataset from GitHub repository."""
        try:
            # For now, we'll implement a basic GitHub ingestion
            # In practice, you'd need to handle different GitHub structures
            logger.warning(f"GitHub ingestion not fully implemented for {source.name}")
            
            # Placeholder implementation
            data = []
            metadata = {
                "total_items": 0,
                "columns": [],
                "source_type": "github",
                "source_path": source.source_path
            }
            
            provenance = {
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "source": source.source_path,
                "license": source.license,
                "description": source.description
            }
            
            return IngestedDataset(
                name=source.name,
                data=data,
                source=source,
                metadata=metadata,
                provenance=provenance
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest GitHub dataset {source.name}: {e}")
            raise
    
    def _ingest_local(self, source: DatasetSource) -> IngestedDataset:
        """Ingest dataset from local file."""
        try:
            file_path = Path(source.source_path)
            
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            elif file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Apply column mapping if specified
            if source.columns:
                data = self._apply_column_mapping(data, source.columns)
            
            # Apply filters if specified
            if source.filters:
                data = self._apply_filters(data, source.filters)
            
            metadata = {
                "total_items": len(data),
                "columns": list(data[0].keys()) if data else [],
                "source_type": "local",
                "source_path": str(file_path)
            }
            
            provenance = {
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "source": str(file_path),
                "license": source.license,
                "description": source.description
            }
            
            return IngestedDataset(
                name=source.name,
                data=data,
                source=source,
                metadata=metadata,
                provenance=provenance
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest local dataset {source.name}: {e}")
            raise
    
    def _ingest_arxiv(self, source: DatasetSource) -> IngestedDataset:
        """Ingest dataset from arXiv paper."""
        # Placeholder for arXiv ingestion
        logger.warning(f"arXiv ingestion not implemented for {source.name}")
        return IngestedDataset(
            name=source.name,
            data=[],
            source=source,
            metadata={"total_items": 0, "columns": [], "source_type": "arxiv"},
            provenance={"ingestion_timestamp": pd.Timestamp.now().isoformat()}
        )
    
    def _apply_column_mapping(self, data: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Apply column mapping to standardize field names."""
        mapped_data = []
        
        for item in data:
            mapped_item = {}
            for standard_name, source_name in mapping.items():
                if source_name in item:
                    mapped_item[standard_name] = item[source_name]
                else:
                    # Keep original if mapping not found
                    mapped_item[source_name] = item.get(source_name)
            
            # Add any unmapped fields
            for key, value in item.items():
                if key not in mapping.values():
                    mapped_item[key] = value
            
            mapped_data.append(mapped_item)
        
        return mapped_data
    
    def _apply_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to the dataset."""
        filtered_data = []
        
        for item in data:
            include = True
            
            for field, condition in filters.items():
                if field not in item:
                    include = False
                    break
                
                value = item[field]
                
                if isinstance(condition, dict):
                    # Handle range conditions
                    if "min" in condition and value < condition["min"]:
                        include = False
                        break
                    if "max" in condition and value > condition["max"]:
                        include = False
                        break
                    if "values" in condition and value not in condition["values"]:
                        include = False
                        break
                elif isinstance(condition, list):
                    # Handle list conditions
                    if value not in condition:
                        include = False
                        break
                else:
                    # Handle exact match
                    if value != condition:
                        include = False
                        break
            
            if include:
                filtered_data.append(item)
        
        return filtered_data
    
    def save_ingested_dataset(self, dataset: IngestedDataset, output_dir: Path):
        """Save ingested dataset to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_file = output_dir / f"{dataset.name}_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(dataset.data, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata_file = output_dir / f"{dataset.name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset.metadata, f, indent=2, ensure_ascii=False)
        
        # Save provenance
        provenance_file = output_dir / f"{dataset.name}_provenance.json"
        with open(provenance_file, 'w', encoding='utf-8') as f:
            json.dump(dataset.provenance, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved ingested dataset {dataset.name} to {output_dir}")
    
    def load_ingested_dataset(self, name: str, input_dir: Path) -> IngestedDataset:
        """Load previously ingested dataset from disk."""
        data_file = input_dir / f"{name}_data.json"
        metadata_file = input_dir / f"{name}_metadata.json"
        provenance_file = input_dir / f"{name}_provenance.json"
        
        if not all(f.exists() for f in [data_file, metadata_file, provenance_file]):
            raise FileNotFoundError(f"Dataset {name} not found in {input_dir}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        with open(provenance_file, 'r', encoding='utf-8') as f:
            provenance = json.load(f)
        
        # Reconstruct source (simplified)
        source = DatasetSource(
            name=name,
            source_type=metadata.get("source_type", "unknown"),
            source_path=metadata.get("source_path", "")
        )
        
        return IngestedDataset(
            name=name,
            data=data,
            source=source,
            metadata=metadata,
            provenance=provenance
        )
