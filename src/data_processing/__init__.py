"""Data processing utilities for rejection detection datasets."""

from .processor import DataProcessor, ProcessingConfig
from .cli import main

__version__ = "0.1.0"
__all__ = ["DataProcessor", "ProcessingConfig", "main"]
