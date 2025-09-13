"""Tests for the rejection_detection package."""

import pytest
import torch
from rejection_detection import (
    main, 
    RejectionDetector,
    MultiHeadClassifier,
    MultiHeadLoss,
    HEAD_CONFIGS,
    get_head_config,
    get_label_to_id_mapping,
    create_sample_data,
)


def test_main():
    """Test the main function."""
    # This is a placeholder test - replace with actual tests
    assert main is not None


def test_rejection_detector_init():
    """Test RejectionDetector initialization."""
    detector = RejectionDetector()
    assert detector is not None


def test_rejection_detector_detect():
    """Test rejection detection functionality."""
    detector = RejectionDetector()
    
    # Test rejection detection
    assert detector.detect("I reject this proposal") is True
    assert detector.detect("REJECT this request") is True
    assert detector.detect("I deny your application") is True
    assert detector.detect("We refuse to proceed") is True
    assert detector.detect("I decline the offer") is True
    assert detector.detect("This is not a rejection") is False
    assert detector.detect("This is an acceptance") is False
    assert detector.detect("") is False


def test_taxonomies():
    """Test taxonomy definitions."""
    # Test head configurations
    assert len(HEAD_CONFIGS) == 5
    assert "head_a" in HEAD_CONFIGS
    assert "head_b_a" in HEAD_CONFIGS
    assert "head_b_b" in HEAD_CONFIGS
    assert "head_c" in HEAD_CONFIGS
    assert "head_d" in HEAD_CONFIGS
    
    # Test head A configuration
    head_a_config = get_head_config("head_a")
    assert head_a_config.num_classes == 6
    assert head_a_config.head_type == "classification"
    assert "REFUSAL.DIRECT" in head_a_config.labels
    assert "COMPLY.BENIGN" in head_a_config.labels
    
    # Test head C configuration (multilabel)
    head_c_config = get_head_config("head_c")
    assert head_c_config.num_classes == 26
    assert head_c_config.head_type == "multilabel"
    assert "weapons" in head_c_config.labels
    assert "other" in head_c_config.labels


def test_label_mappings():
    """Test label to ID mappings."""
    head_a_mapping = get_label_to_id_mapping("head_a")
    assert "REFUSAL.DIRECT" in head_a_mapping
    assert head_a_mapping["REFUSAL.DIRECT"] == 0
    assert head_a_mapping["COMPLY.BENIGN"] == 3


def test_multi_head_classifier():
    """Test multi-head classifier initialization."""
    model = MultiHeadClassifier(
        model_name="bert-base-uncased",
        freeze_encoder=True,
    )
    
    assert model is not None
    assert len(model.heads) == 5
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask)
    
    # Check that all heads produce outputs
    for head_name in HEAD_CONFIGS.keys():
        assert head_name in outputs
        head_config = get_head_config(head_name)
        assert outputs[head_name].shape == (batch_size, head_config.num_classes)


def test_multi_head_loss():
    """Test multi-head loss function."""
    criterion = MultiHeadLoss()
    
    # Create dummy outputs and labels
    batch_size = 2
    outputs = {}
    labels = {}
    
    for head_name, head_config in HEAD_CONFIGS.items():
        if head_config.head_type == "multilabel":
            outputs[head_name] = torch.rand(batch_size, head_config.num_classes)
            labels[head_name] = torch.randint(0, 2, (batch_size, head_config.num_classes)).float()
        else:
            outputs[head_name] = torch.rand(batch_size, head_config.num_classes)
            labels[head_name] = torch.randint(0, head_config.num_classes, (batch_size,))
    
    losses = criterion(outputs, labels)
    
    # Check that all heads have losses
    for head_name in HEAD_CONFIGS.keys():
        assert head_name in losses
        assert losses[head_name].item() >= 0
    
    assert "total" in losses
    assert losses["total"].item() >= 0


def test_sample_data_creation():
    """Test sample data creation."""
    sample_data = create_sample_data(num_samples=10)
    
    assert len(sample_data) == 10
    
    # Check sample structure
    sample = sample_data[0]
    assert "prompt" in sample
    assert "response" in sample
    assert "head_a" in sample
    assert "head_c" in sample
    assert "head_d" in sample
    
    # Check head_a values
    head_a_config = get_head_config("head_a")
    assert sample["head_a"] in head_a_config.labels
    
    # Check head_c is a list
    assert isinstance(sample["head_c"], list)
    
    # Check head_d is a dict
    assert isinstance(sample["head_d"], dict)
    assert "prompt_harmful" in sample["head_d"]
    assert "response_harmful" in sample["head_d"]
    assert "response_refusal" in sample["head_d"]
