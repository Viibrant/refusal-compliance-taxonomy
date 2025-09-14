"""Tests for the data_utils module."""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from rejection_detection.data_utils import (
    RejectionDetectionDataset,
    load_data_from_json as load_data,
    save_data,
    split_data,
    create_sample_data
)
from rejection_detection.taxonomies import get_head_configs, get_head_config


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_b_a": "STYLE.DIRECT",
                "head_b_b": None,
                "head_c_a": ["weapons"],
                "head_c_b": ["technology"],
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            },
            {
                "prompt": "What is the weather like?",
                "response": "I don't have access to real-time weather data.",
                "head_a": "REFUSAL.CAPABILITY",
                "head_b_a": "STYLE.CLARIFY",
                "head_b_b": None,
                "head_c_a": [],
                "head_c_b": ["technology"],
                "head_d": {"prompt_harmful": False, "response_harmful": False, "response_refusal": True}
            }
        ]
    
    @pytest.fixture
    def sample_data_file(self, sample_data):
        """Create a sample data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            f.flush()  # Ensure data is written
            temp_file = Path(f.name)
        
        yield temp_file
        
        # Clean up
        if temp_file.exists():
            temp_file.unlink()
    
    def test_load_data(self, sample_data_file):
        """Test loading data from file."""
        data = load_data(sample_data_file)
        
        assert len(data) == 2
        assert "prompt" in data[0]
        assert "response" in data[0]
        assert "head_a" in data[0]
        assert "head_c_a" in data[0]
        assert "head_c_b" in data[0]
        assert "head_d" in data[0]
    
    def test_load_data_nonexistent_file(self):
        """Test loading data from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent.json"))
    
    def test_load_data_invalid_json(self):
        """Test loading data from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            with pytest.raises(json.JSONDecodeError):
                load_data(Path(f.name))
    
    def test_save_data(self, sample_data_file):
        """Test saving data to file."""
        data = load_data(sample_data_file)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)
        
        save_data(data, output_file)
        
        # Verify data was saved correctly
        saved_data = load_data(output_file)
        assert len(saved_data) == len(data)
        assert saved_data[0]["prompt"] == data[0]["prompt"]
    
    def test_save_data_with_numpy_types(self):
        """Test saving data with numpy types."""
        import numpy as np
        
        data = [
            {
                "prompt": "Test prompt",
                "response": "Test response",
                "head_a": "REFUSAL.DIRECT",
                "head_c_a": ["weapons"],
                "head_c_b": ["technology"],
                "head_d": {
                    "prompt_harmful": np.bool_(True),
                    "response_harmful": np.bool_(False),
                    "response_refusal": np.bool_(True)
                }
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)
        
        save_data(data, output_file)
        
        # Verify data was saved correctly
        saved_data = load_data(output_file)
        assert saved_data[0]["head_d"]["prompt_harmful"] is True
        assert saved_data[0]["head_d"]["response_harmful"] is False
        assert saved_data[0]["head_d"]["response_refusal"] is True


class TestDataSplitting:
    """Test data splitting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {"id": i, "prompt": f"prompt_{i}", "response": f"response_{i}"}
            for i in range(100)
        ]
    
    def test_split_data_default_ratios(self, sample_data):
        """Test data splitting with default ratios."""
        train, val, test = split_data(sample_data)
        
        assert len(train) == 80  # 80% (default ratio)
        assert len(val) == 10    # 10%
        assert len(test) == 10   # 10%
        
        # Check that all data is accounted for
        assert len(train) + len(val) + len(test) == len(sample_data)
    
    def test_split_data_custom_ratios(self, sample_data):
        """Test data splitting with custom ratios."""
        train, val, test = split_data(sample_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        
        assert len(train) == 80  # 80%
        assert len(val) == 10    # 10%
        assert len(test) == 10   # 10%
    
    def test_split_data_small_dataset(self):
        """Test data splitting with small dataset."""
        small_data = [{"id": i} for i in range(5)]
        
        train, val, test = split_data(small_data)
        
        # Should handle small datasets gracefully
        assert len(train) + len(val) + len(test) == len(small_data)
        assert len(train) >= 0
        assert len(val) >= 0
        assert len(test) >= 0
    
    def test_split_data_with_seed(self, sample_data):
        """Test data splitting with fixed seed."""
        train1, val1, test1 = split_data(sample_data, random_seed=42)
        train2, val2, test2 = split_data(sample_data, random_seed=42)
        
        # Should produce same splits with same seed
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)


class TestSampleDataCreation:
    """Test sample data creation functionality."""
    
    def test_create_sample_data_default(self):
        """Test creating sample data with default parameters."""
        data = create_sample_data()
        
        assert len(data) == 100  # Default num_samples
        assert "prompt" in data[0]
        assert "response" in data[0]
        assert "head_a" in data[0]
        assert "head_c_a" in data[0]
        assert "head_c_b" in data[0]
        assert "head_d" in data[0]
    
    def test_create_sample_data_custom_size(self):
        """Test creating sample data with custom size."""
        data = create_sample_data(num_samples=50)
        
        assert len(data) == 50
    
    def test_create_sample_data_structure(self):
        """Test that sample data has correct structure."""
        data = create_sample_data(num_samples=10)
        
        for item in data:
            # Check required fields
            assert "prompt" in item
            assert "response" in item
            assert "head_a" in item
            assert "head_c_a" in item
            assert "head_c_b" in item
            assert "head_d" in item
            
            # Check head_a is valid
            head_a_config = get_head_config("head_a")
            assert item["head_a"] in head_a_config.class_names
            
            # Check head_c_a and head_c_b are lists
            assert isinstance(item["head_c_a"], list)
            assert isinstance(item["head_c_b"], list)
            
            # Check head_d is a dict with correct keys
            assert isinstance(item["head_d"], dict)
            assert "prompt_harmful" in item["head_d"]
            assert "response_harmful" in item["head_d"]
            assert "response_refusal" in item["head_d"]
            
            # Check boolean values (numpy booleans are also valid)
            assert isinstance(item["head_d"]["prompt_harmful"], (bool, np.bool_))
            assert isinstance(item["head_d"]["response_harmful"], (bool, np.bool_))
            assert isinstance(item["head_d"]["response_refusal"], (bool, np.bool_))


class TestRejectionDetectionDataset:
    """Test RejectionDetectionDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_b_a": "STYLE.DIRECT",
                "head_b_b": None,
                "head_c_a": ["weapons"],
                "head_c_b": ["technology"],
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            },
            {
                "prompt": "What is the weather like?",
                "response": "I don't have access to real-time weather data.",
                "head_a": "REFUSAL.CAPABILITY",
                "head_b_a": "STYLE.CLARIFY",
                "head_b_b": None,
                "head_c_a": [],
                "head_c_b": ["technology"],
                "head_d": {"prompt_harmful": False, "response_harmful": False, "response_refusal": True}
            }
        ]
    
    @pytest.fixture
    def sample_data_file(self, sample_data):
        """Create a sample data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            f.flush()  # Ensure data is written
            temp_file = Path(f.name)
        
        yield temp_file
        
        # Clean up
        if temp_file.exists():
            temp_file.unlink()
    
    @patch('rejection_detection.data_utils.AutoTokenizer.from_pretrained')
    def test_dataset_initialization(self, mock_tokenizer, sample_data):
        """Test dataset initialization."""
        mock_tokenizer.return_value = Mock()
        
        dataset = RejectionDetectionDataset(
            data=sample_data,
            tokenizer=mock_tokenizer.return_value,
            max_length=128
        )
        
        assert len(dataset) == 2
        assert dataset.tokenizer is not None
        assert dataset.max_length == 128
    
    @patch('rejection_detection.data_utils.AutoTokenizer.from_pretrained')
    def test_dataset_getitem(self, mock_tokenizer, sample_data):
        """Test dataset __getitem__ method."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        dataset = RejectionDetectionDataset(
            data=sample_data,
            tokenizer=mock_tokenizer_instance,
            max_length=128
        )
        
        # Get first item
        item = dataset[0]
        
        # Check structure
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        # Check labels structure
        labels = item["labels"]
        head_configs = get_head_configs()
        for head_name in head_configs.keys():
            assert head_name in labels
    
    @patch('rejection_detection.data_utils.AutoTokenizer.from_pretrained')
    def test_dataset_with_style_heads(self, mock_tokenizer, sample_data):
        """Test dataset with style heads included."""
        mock_tokenizer.return_value = Mock()
        
        dataset = RejectionDetectionDataset(
            data=sample_data,
            tokenizer=mock_tokenizer.return_value,
            max_length=128,
            include_style_heads=True
        )
        
        assert dataset.include_style_heads is True
    
    @patch('rejection_detection.data_utils.AutoTokenizer.from_pretrained')
    def test_dataset_without_style_heads(self, mock_tokenizer, sample_data):
        """Test dataset without style heads."""
        mock_tokenizer.return_value = Mock()
        
        dataset = RejectionDetectionDataset(
            data=sample_data,
            tokenizer=mock_tokenizer.return_value,
            max_length=128,
            include_style_heads=False
        )
        
        assert dataset.include_style_heads is False


class TestDataValidation:
    """Test data validation functionality."""
    
    def test_save_data_with_numpy_types(self):
        """Test saving data with numpy types."""
        import numpy as np
        
        data = [
            {
                "prompt": "Test prompt",
                "response": "Test response",
                "head_a": "REFUSAL.DIRECT",
                "head_c_a": ["weapons"],
                "head_c_b": ["technology"],
                "head_d": {
                    "prompt_harmful": np.bool_(True),
                    "response_harmful": np.bool_(False),
                    "response_refusal": np.bool_(True)
                }
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)
        
        save_data(data, output_file)
        
        # Verify data was saved correctly
        saved_data = load_data(output_file)
        assert saved_data[0]["head_d"]["prompt_harmful"] is True
        assert saved_data[0]["head_d"]["response_harmful"] is False
        assert saved_data[0]["head_d"]["response_refusal"] is True


class TestDataIntegration:
    """Integration tests for data utilities."""
    
    def test_full_data_pipeline(self):
        """Test full data pipeline from creation to loading."""
        # Create sample data
        data = create_sample_data(num_samples=20)
        
        # Save data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)
        
        save_data(data, output_file)
        
        # Load data
        loaded_data = load_data(output_file)
        
        # Verify data integrity
        assert len(loaded_data) == len(data)
        assert loaded_data[0]["prompt"] == data[0]["prompt"]
        assert loaded_data[0]["head_a"] == data[0]["head_a"]
        
        # Split data
        train, val, test = split_data(loaded_data)
        
        # Verify splits
        assert len(train) + len(val) + len(test) == len(loaded_data)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
    
    @patch('rejection_detection.data_utils.AutoTokenizer.from_pretrained')
    def test_dataset_with_real_data(self, mock_tokenizer):
        """Test dataset with real data pipeline."""
        # Create sample data
        data = create_sample_data(num_samples=10)
        
        # Save data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)
        
        save_data(data, output_file)
        
        # Mock tokenizer
        mock_tokenizer.return_value = Mock()
        
        # Create dataset
        dataset = RejectionDetectionDataset(
            data=data,
            tokenizer=mock_tokenizer.return_value,
            max_length=128
        )
        
        assert len(dataset) == 10
        assert dataset.tokenizer is not None
