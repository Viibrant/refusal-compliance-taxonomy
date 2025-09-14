"""Tests for new functionality including dynamic head configuration and metric fixes."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from rejection_detection.taxonomies import (
    OutcomeType, HarmCategory, HarmlessCategory, get_head_configs, get_head_config,
    get_outcome_definitions, get_minimal_examples, get_harmless_category_definitions
)
from rejection_detection.model import MultiHeadClassifier, MultiHeadLoss
from rejection_detection.training import MultiHeadTrainer
from rejection_detection.data_utils import RejectionDetectionDataset


class TestDynamicHeadConfiguration:
    """Test dynamic head configuration loading."""
    
    def test_model_dynamically_loads_head_configs(self):
        """Test that model dynamically loads head configurations."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Check that all expected heads are present
                expected_heads = ["head_a", "head_b", "head_c_a", "head_c_b", "head_d"]
                for head_name in expected_heads:
                    assert head_name in model.heads
                
                # Check that head configurations match taxonomy
                head_configs = get_head_configs()
                for head_name, head_config in head_configs.items():
                    assert head_name in model.heads
                    # The head should have the correct number of output classes
                    head_module = model.heads[head_name]
                    if hasattr(head_module, 'out_features'):
                        assert head_module.out_features == head_config.num_classes
                    else:
                        # For Sequential modules, check the last linear layer
                        last_linear = None
                        for layer in reversed(head_module):
                            if hasattr(layer, 'out_features'):
                                last_linear = layer
                                break
                        assert last_linear is not None
                        assert last_linear.out_features == head_config.num_classes
    
    def test_model_handles_new_compliance_types(self):
        """Test that model handles new compliance types correctly."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Check that head_a has the correct number of classes (12 with new compliance types)
                head_a_config = get_head_config("head_a")
                assert head_a_config.num_classes == 12
                
                # Check that new compliance types are in the class names
                assert "COMPLY.CONDITIONAL" in head_a_config.class_names
                assert "COMPLY.EDUCATIONAL" in head_a_config.class_names
                assert "COMPLY.REDIRECTED" in head_a_config.class_names
                assert "COMPLY.PARTIAL_SAFE" in head_a_config.class_names
    
    def test_model_handles_harmless_categories(self):
        """Test that model handles harmless categories correctly."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Check that head_c_b has the correct number of classes (20 harmless categories)
                head_c_b_config = get_head_config("head_c_b")
                assert head_c_b_config.num_classes == 20
                
                # Check that harmless categories are in the class names
                harmless_categories = [cat.value for cat in HarmlessCategory]
                for category in harmless_categories:
                    assert category in head_c_b_config.class_names


class TestNewComplianceTypesIntegration:
    """Test integration of new compliance types."""
    
    def test_new_compliance_types_in_model_predictions(self):
        """Test that model can handle new compliance types in predictions."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_encoder.return_value = Mock(
                    last_hidden_state=torch.randn(2, 128, 768),
                    pooler_output=torch.randn(2, 768)
                )
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Create sample input
                input_ids = torch.randint(0, 1000, (2, 128))
                attention_mask = torch.ones(2, 128)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Check that head_a output has correct shape (12 classes)
                assert outputs["head_a"].shape == (2, 12)
                
                # Check that we can get predictions for new compliance types
                head_a_config = get_head_config("head_a")
                new_compliance_indices = []
                for outcome_type in [OutcomeType.COMPLY_CONDITIONAL, OutcomeType.COMPLY_EDUCATIONAL, 
                                   OutcomeType.COMPLY_REDIRECTED, OutcomeType.COMPLY_PARTIAL_SAFE]:
                    if outcome_type.value in head_a_config.class_names:
                        idx = head_a_config.class_names.index(outcome_type.value)
                        new_compliance_indices.append(idx)
                
                # Should have found all new compliance types
                assert len(new_compliance_indices) == 4
    
    def test_new_compliance_types_in_loss_computation(self):
        """Test that loss computation works with new compliance types."""
        criterion = MultiHeadLoss()
        
        # Create outputs and labels with new compliance types
        outputs = {
            "head_a": torch.randn(2, 12),  # 12 classes including new compliance types
        }
        labels = {
            "head_a": torch.tensor([5, 6]),  # Use indices for new compliance types
        }
        
        # Compute loss
        losses = criterion(outputs, labels)
        
        # Check that loss is computed without errors
        assert "head_a" in losses
        assert losses["head_a"].item() >= 0
        assert "total" in losses


class TestHarmlessCategoriesIntegration:
    """Test integration of harmless categories."""
    
    def test_harmless_categories_in_model_predictions(self):
        """Test that model can handle harmless categories in predictions."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_encoder.return_value = Mock(
                    last_hidden_state=torch.randn(2, 128, 768),
                    pooler_output=torch.randn(2, 768)
                )
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Create sample input
                input_ids = torch.randint(0, 1000, (2, 128))
                attention_mask = torch.ones(2, 128)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Check that head_c_b output has correct shape (20 classes)
                assert outputs["head_c_b"].shape == (2, 20)
                
                # Check that we can get predictions for harmless categories
                head_c_b_config = get_head_config("head_c_b")
                harmless_categories = [cat.value for cat in HarmlessCategory]
                for category in harmless_categories:
                    assert category in head_c_b_config.class_names
    
    def test_harmless_categories_in_loss_computation(self):
        """Test that loss computation works with harmless categories."""
        criterion = MultiHeadLoss()
        
        # Create outputs and labels for harmless categories
        outputs = {
            "head_c_b": torch.sigmoid(torch.randn(2, 20)),  # 20 harmless categories
        }
        labels = {
            "head_c_b": torch.randint(0, 2, (2, 20)).float(),  # Binary labels
        }
        
        # Compute loss
        losses = criterion(outputs, labels)
        
        # Check that loss is computed without errors
        assert "head_c_b" in losses
        assert losses["head_c_b"].item() >= 0
        assert "total" in losses


class TestDataUtilsIntegration:
    """Test data utils integration with new taxonomy structure."""
    
    def test_dataset_handles_new_structure(self):
        """Test that dataset handles the new head_c_a and head_c_b structure."""
        # Create sample data with new structure
        sample_data = [
            {
                "prompt": "What are the health benefits of exercise?",
                "response": "Exercise has numerous health benefits including improved cardiovascular health.",
                "head_a": "COMPLY.BENIGN",
                "head_b": "STYLE.DIRECT_ANSWER",
                "head_c_a": [],  # No harm categories
                "head_c_b": ["health_medicine", "education"],  # Harmless categories
                "head_d": {"prompt_harmful": False, "response_harmful": False, "response_refusal": False}
            },
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_b": "STYLE.DIRECT",
                "head_c_a": ["weapons"],  # Harm categories
                "head_c_b": [],  # No harmless categories
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            }
        ]
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128)
        }
        
        # Create dataset
        dataset = RejectionDetectionDataset(
            sample_data,
            mock_tokenizer,
            max_length=128
        )
        
        # Test that dataset can be created without errors
        assert len(dataset) == 2
        
        # Test that we can get an item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        # Check that labels have the expected structure
        labels = item["labels"]
        expected_label_fields = ["head_a", "head_c_a", "head_c_b", "head_d"]
        for field in expected_label_fields:
            assert field in labels


class TestTaxonomyConsistency:
    """Test consistency across the updated taxonomy."""
    
    def test_all_new_compliance_types_have_definitions(self):
        """Test that all new compliance types have definitions."""
        definitions = get_outcome_definitions()
        examples = get_minimal_examples()
        
        new_compliance_types = [
            OutcomeType.COMPLY_CONDITIONAL,
            OutcomeType.COMPLY_EDUCATIONAL, 
            OutcomeType.COMPLY_REDIRECTED,
            OutcomeType.COMPLY_PARTIAL_SAFE
        ]
        
        for compliance_type in new_compliance_types:
            assert compliance_type in definitions
            assert compliance_type.value in examples
            assert len(definitions[compliance_type]) > 0
            assert len(examples[compliance_type.value]) > 0
    
    def test_all_harmless_categories_have_definitions(self):
        """Test that all harmless categories have definitions."""
        definitions = get_harmless_category_definitions()
        
        for category in HarmlessCategory:
            assert category in definitions
            assert len(definitions[category]) > 0
    
    def test_head_configurations_are_consistent(self):
        """Test that head configurations are consistent with enums."""
        configs = get_head_configs()
        
        # Check head_a (OutcomeType)
        head_a_config = configs["head_a"]
        outcome_values = [e.value for e in OutcomeType]
        assert len(head_a_config.class_names) == len(outcome_values)
        for class_name in head_a_config.class_names:
            assert class_name in outcome_values
        
        # Check head_c_a (HarmCategory)
        head_c_a_config = configs["head_c_a"]
        harm_values = [e.value for e in HarmCategory]
        assert len(head_c_a_config.class_names) == len(harm_values)
        for class_name in head_c_a_config.class_names:
            assert class_name in harm_values
        
        # Check head_c_b (HarmlessCategory)
        head_c_b_config = configs["head_c_b"]
        harmless_values = [e.value for e in HarmlessCategory]
        assert len(head_c_b_config.class_names) == len(harmless_values)
        for class_name in head_c_b_config.class_names:
            assert class_name in harmless_values
    
    def test_no_overlap_between_harm_and_harmless_categories(self):
        """Test that there's no overlap between harm and harmless categories (except 'other')."""
        harm_categories = [cat.value for cat in HarmCategory]
        harmless_categories = [cat.value for cat in HarmlessCategory]
        
        overlap = set(harm_categories) & set(harmless_categories)
        # Only 'other' should overlap, which is intentional
        assert overlap == {"other"}, f"Unexpected overlap found: {overlap}"


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""
    
    def test_existing_functionality_still_works(self):
        """Test that existing functionality still works with new structure."""
        # Test that we can still get head configurations
        configs = get_head_configs()
        assert "head_a" in configs
        assert "head_b" in configs
        assert "head_d" in configs
        
        # Test that we can still get individual head configs
        head_a = get_head_config("head_a")
        assert head_a is not None
        assert head_a.head_type == "classification"
        
        head_d = get_head_config("head_d")
        assert head_d is not None
        assert head_d.head_type == "boolean"
    
    def test_model_can_handle_old_and_new_structures(self):
        """Test that model can handle both old and new data structures."""
        with patch('rejection_detection.model.AutoModel.from_pretrained') as mock_model:
            with patch('rejection_detection.model.AutoConfig.from_pretrained') as mock_config:
                # Mock the config and model
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Create a properly structured mock encoder
                mock_encoder = Mock()
                mock_encoder.parameters.return_value = [Mock(), Mock()]
                
                # Mock the encoder.encoder.layer structure for layer freezing
                mock_layers = []
                for _ in range(12):  # 12 BERT layers
                    mock_layer = Mock()
                    mock_layer.parameters.return_value = [Mock(), Mock()]
                    mock_layers.append(mock_layer)
                mock_encoder.encoder.layer = mock_layers
                
                # Mock embeddings and pooler
                mock_encoder.embeddings = Mock()
                mock_encoder.embeddings.parameters.return_value = [Mock(), Mock()]
                mock_encoder.pooler = Mock()
                mock_encoder.pooler.parameters.return_value = [Mock(), Mock()]
                
                mock_encoder.return_value = Mock(
                    last_hidden_state=torch.randn(2, 128, 768),
                    pooler_output=torch.randn(2, 768)
                )
                mock_model.return_value = mock_encoder
                
                # Create model
                model = MultiHeadClassifier(
                    model_name="bert-base-uncased",
                    freeze_encoder=True
                )
                
                # Test with new structure (head_c_a and head_c_b)
                input_ids = torch.randint(0, 1000, (2, 128))
                attention_mask = torch.ones(2, 128)
                
                outputs = model(input_ids, attention_mask)
                
                # Should have all expected heads
                expected_heads = ["head_a", "head_b", "head_c_a", "head_c_b", "head_d"]
                for head_name in expected_heads:
                    assert head_name in outputs
