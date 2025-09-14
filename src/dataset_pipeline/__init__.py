"""Dataset processing pipeline for rejection detection training data."""

from .pipeline import DatasetPipeline, PipelineConfig
from .ingestion import DatasetIngester, IngestedDataset
from .generation import ResponseGenerator, GenerationConfig
from .labeling import CAIJudge, LabelingConfig
from .quality import QualityController, AuditReport
from .config import DatasetSource
from .cli import main

__version__ = "0.1.0"
__all__ = [
    "DatasetPipeline", 
    "PipelineConfig",
    "DatasetIngester", 
    "IngestedDataset",
    "ResponseGenerator", 
    "GenerationConfig",
    "CAIJudge", 
    "LabelingConfig",
    "QualityController", 
    "AuditReport",
    "DatasetSource",
    "main"
]
