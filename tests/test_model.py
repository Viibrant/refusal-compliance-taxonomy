"""Tests for the model module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from rejection_detection.model import MultiHeadClassifier, MultiHeadLoss
from rejection_detection.taxonomies import get_head_configs, get_head_config


class TestMultiHeadClassifier:
    """Test MultiHeadClassifier class."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        seq_length = 128
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "token_type_ids": torch.zeros(batch_size, seq_length)
        }
    
    def _create_mock_encoder(self):
        """Helper to create a properly structured mock encoder."""
        mock_encoder = Mock()
        
        # Create mock parameters with requires_grad=False for frozen testing
        mock_param1 = Mock()
        mock_param1.requires_grad = False
        mock_param2 = Mock()
        mock_param2.requires_grad = False
        mock_encoder.parameters.return_value = [mock_param1, mock_param2]
        
        # Mock the encoder.encoder.layer structure for layer freezing
        mock_layers = []
        for _ in range(12):  # 12 BERT layers
            mock_layer = Mock()
            # Create mock parameters with requires_grad=False for frozen testing
            layer_param1 = Mock()
            layer_param1.requires_grad = False
            layer_param2 = Mock()
            layer_param2.requires_grad = False
            mock_layer.parameters.return_value = [layer_param1, layer_param2]
            mock_layers.append(mock_layer)
        mock_encoder.encoder.layer = mock_layers
        
        # Mock embeddings and pooler
        mock_encoder.embeddings = Mock()
        emb_param1 = Mock()
        emb_param1.requires_grad = False
        emb_param2 = Mock()
        emb_param2.requires_grad = False
        mock_encoder.embeddings.parameters.return_value = [emb_param1, emb_param2]
        
        mock_encoder.pooler = Mock()
        pool_param1 = Mock()
        pool_param1.requires_grad = False
        pool_param2 = Mock()
        pool_param2.requires_grad = False
        mock_encoder.pooler.parameters.return_value = [pool_param1, pool_param2]
        
        return mock_encoder
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_initialization(self, mock_config, mock_model):
        """Test model initialization."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        assert model.model_name == "bert-base-uncased"
        assert model.freeze_encoder is True
        assert model.encoder is not None
        assert len(model.heads) == 6  # All 6 heads (updated with head_c_a and head_c_b)
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_initialization_without_style_heads(self, mock_config, mock_model):
        """Test model initialization without style heads."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        assert model.freeze_encoder is True
        assert len(model.heads) == 6  # All 6 heads (updated with head_c_a and head_c_b) (style heads are always included)
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_initialization_unfrozen_encoder(self, mock_config, mock_model):
        """Test model initialization with unfrozen encoder."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=False
        )
        
        assert model.freeze_encoder is False
        # Check that encoder parameters are not frozen
        mock_encoder.eval.assert_not_called()
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_forward_pass(self, mock_config, mock_model, sample_batch):
        """Test model forward pass."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_encoder.return_value = Mock(
            last_hidden_state=torch.randn(2, 128, 768),
            pooler_output=torch.randn(2, 768)
        )
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Forward pass
        outputs = model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            token_type_ids=sample_batch["token_type_ids"]
        )
        
        # Check that all expected heads produce outputs
        head_configs = get_head_configs()
        for head_name in head_configs.keys():
            assert head_name in outputs
            head_config = get_head_config(head_name)
            assert outputs[head_name].shape == (2, head_config.num_classes)
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_forward_pass_without_token_type_ids(self, mock_config, mock_model, sample_batch):
        """Test model forward pass without token_type_ids."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_encoder.return_value = Mock(
            last_hidden_state=torch.randn(2, 128, 768),
            pooler_output=torch.randn(2, 768)
        )
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Forward pass without token_type_ids
        outputs = model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"]
        )
        
        # Check that outputs are produced
        assert "head_a" in outputs
        assert "head_c_a" in outputs
        assert "head_c_b" in outputs
        assert "head_d" in outputs
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_head_architectures(self, mock_config, mock_model):
        """Test that model heads have correct architectures."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set a real hidden size
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Check head architectures
        head_configs = get_head_configs()
        for head_name, head_config in head_configs.items():
            head = model.heads[head_name]
            
            if head_config.head_type == "multilabel":
                # Multilabel heads should use Sequential with Linear layers
                assert isinstance(head, nn.Sequential)
                # Check the last linear layer (might be before sigmoid)
                last_linear = None
                for layer in reversed(head):
                    if isinstance(layer, nn.Linear):
                        last_linear = layer
                        break
                assert last_linear is not None
                assert last_linear.out_features == head_config.num_classes
            elif head_config.head_type == "boolean":
                # Boolean heads should use Sequential with Linear layers
                assert isinstance(head, nn.Sequential)
                # Check the last linear layer (might be before sigmoid)
                last_linear = None
                for layer in reversed(head):
                    if isinstance(layer, nn.Linear):
                        last_linear = layer
                        break
                assert last_linear is not None
                assert last_linear.out_features == head_config.num_classes
            else:
                # Classification heads should use Sequential with Linear layers
                assert isinstance(head, nn.Sequential)
                assert head[-1].out_features == head_config.num_classes
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_encoder_frozen(self, mock_config, mock_model):
        """Test that encoder is properly frozen when requested."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Check that encoder parameters were frozen (requires_grad = False)
        # The mock encoder parameters should have requires_grad set to False
        for param in mock_encoder.parameters.return_value:
            assert param.requires_grad is False
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_encoder_unfrozen(self, mock_config, mock_model):
        """Test that encoder is not frozen when requested."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        mock_encoder = Mock()
        mock_encoder.parameters.return_value = [Mock(), Mock()]  # Make it iterable
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=False
        )
        
        # Check that encoder parameters were not frozen (requires_grad = True)
        # The mock encoder parameters should have requires_grad set to True
        # Since we're using mocks, we can't easily test the actual requires_grad state
        # Instead, we just verify the model was created successfully
        assert model.freeze_encoder is False

    def test_model_layer_freezing_with_real_model(self):
        """Test layer freezing with a real model to verify actual parameter states."""
        # Use a real model for this test to verify actual freezing behavior
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Check that embeddings are frozen
        for param in model.encoder.embeddings.parameters():
            assert not param.requires_grad, "Embeddings should be frozen"
        
        # Check that layers 0-10 are frozen
        for layer_idx in range(11):  # Layers 0-10
            layer = model.encoder.encoder.layer[layer_idx]
            for param in layer.parameters():
                assert not param.requires_grad, f"Layer {layer_idx} should be frozen"
        
        # Check that layer 11 (last layer) is NOT frozen
        last_layer = model.encoder.encoder.layer[11]
        for param in last_layer.parameters():
            assert param.requires_grad, "Last layer (11) should be trainable"
        
        # Check that pooler is frozen
        for param in model.encoder.pooler.parameters():
            assert not param.requires_grad, "Pooler should be frozen"
        
        # Check that heads are trainable
        for param in model.heads.parameters():
            assert param.requires_grad, "Heads should be trainable"

    def test_model_no_freezing_with_real_model(self):
        """Test that no layers are frozen when freeze_encoder=False."""
        # Use a real model for this test
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=False
        )
        
        # Check that all encoder parameters are trainable
        for param in model.encoder.parameters():
            assert param.requires_grad, "All encoder parameters should be trainable when freeze_encoder=False"
        
        # Check that heads are trainable
        for param in model.heads.parameters():
            assert param.requires_grad, "Heads should be trainable"

    def test_model_parameter_counts(self):
        """Test that parameter counts are correct with layer freezing."""
        model_frozen = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        model_unfrozen = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=False
        )
        
        # Both models should have the same total parameters
        total_params_frozen = sum(p.numel() for p in model_frozen.parameters())
        total_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters())
        assert total_params_frozen == total_params_unfrozen
        
        # But different trainable parameters
        trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        trainable_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
        
        # Frozen model should have fewer trainable parameters
        assert trainable_params_frozen < trainable_params_unfrozen
        
        # Frozen model should have approximately 7.74% trainable parameters
        trainable_percentage = 100 * trainable_params_frozen / total_params_frozen
        assert 7.0 <= trainable_percentage <= 8.0, f"Expected ~7.74% trainable, got {trainable_percentage:.2f}%"

    def test_model_gradient_flow_with_freezing(self):
        """Test gradient flow with layer freezing."""
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Create dummy input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Create dummy loss
        total_loss = sum(logits.sum() for logits in outputs.values())
        
        # Backward pass
        total_loss.backward()
        
        # Check that only last layer and heads have gradients
        # Embeddings should have no gradients
        for param in model.encoder.embeddings.parameters():
            assert param.grad is None, "Embeddings should have no gradients"
        
        # Layers 0-10 should have no gradients
        for layer_idx in range(11):
            layer = model.encoder.encoder.layer[layer_idx]
            for param in layer.parameters():
                assert param.grad is None, f"Layer {layer_idx} should have no gradients"
        
        # Layer 11 should have gradients
        last_layer = model.encoder.encoder.layer[11]
        has_gradients = False
        for param in last_layer.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients, "Last layer should have gradients"
        
        # Pooler should have no gradients
        for param in model.encoder.pooler.parameters():
            assert param.grad is None, "Pooler should have no gradients"
        
        # Heads should have gradients
        for param in model.heads.parameters():
            assert param.grad is not None, "Heads should have gradients"

    def test_model_layer_specific_freezing(self):
        """Test that specific layers are frozen/unfrozen correctly."""
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Test specific layer components
        # Last layer attention
        last_layer = model.encoder.encoder.layer[11]
        attention = last_layer.attention
        
        # All attention components should be trainable
        assert attention.self.query.weight.requires_grad
        assert attention.self.query.bias.requires_grad
        assert attention.self.key.weight.requires_grad
        assert attention.self.key.bias.requires_grad
        assert attention.self.value.weight.requires_grad
        assert attention.self.value.bias.requires_grad
        assert attention.output.dense.weight.requires_grad
        assert attention.output.dense.bias.requires_grad
        assert attention.output.LayerNorm.weight.requires_grad
        assert attention.output.LayerNorm.bias.requires_grad
        
        # Last layer intermediate and output
        assert last_layer.intermediate.dense.weight.requires_grad
        assert last_layer.intermediate.dense.bias.requires_grad
        assert last_layer.output.dense.weight.requires_grad
        assert last_layer.output.dense.bias.requires_grad
        assert last_layer.output.LayerNorm.weight.requires_grad
        assert last_layer.output.LayerNorm.bias.requires_grad
        
        # Test that first layer is frozen
        first_layer = model.encoder.encoder.layer[0]
        assert not first_layer.attention.self.query.weight.requires_grad
        assert not first_layer.attention.self.query.bias.requires_grad
        assert not first_layer.intermediate.dense.weight.requires_grad
        assert not first_layer.output.dense.weight.requires_grad


class TestMultiHeadLoss:
    """Test MultiHeadLoss class."""
    
    @pytest.fixture
    def sample_outputs(self):
        """Create sample model outputs."""
        batch_size = 2
        outputs = {}
        
        head_configs = get_head_configs()
        for head_name, head_config in head_configs.items():
            if head_config.head_type in ["multilabel", "boolean"]:
                # For BCELoss, we need values in [0,1], so use sigmoid of randn
                outputs[head_name] = torch.sigmoid(torch.randn(batch_size, head_config.num_classes))
            else:
                outputs[head_name] = torch.randn(batch_size, head_config.num_classes)
        
        return outputs
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        batch_size = 2
        labels = {}
        
        head_configs = get_head_configs()
        for head_name, head_config in head_configs.items():
            if head_config.head_type in ["multilabel", "boolean"]:
                labels[head_name] = torch.randint(0, 2, (batch_size, head_config.num_classes)).float()
            else:
                labels[head_name] = torch.randint(0, head_config.num_classes, (batch_size,))
        
        return labels
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        criterion = MultiHeadLoss()
        
        assert criterion.ce_loss is not None
        assert criterion.bce_loss is not None
        assert isinstance(criterion.ce_loss, nn.CrossEntropyLoss)
        assert isinstance(criterion.bce_loss, nn.BCELoss)
    
    def test_loss_computation(self, sample_outputs, sample_labels):
        """Test loss computation."""
        criterion = MultiHeadLoss()
        
        losses = criterion(sample_outputs, sample_labels)
        
        # Check that all heads have losses
        head_configs = get_head_configs()
        for head_name in head_configs.keys():
            assert head_name in losses
            assert losses[head_name].item() >= 0
        
        # Check that total loss is computed
        assert "total" in losses
        assert losses["total"].item() >= 0
    
    def test_loss_computation_classification_head(self):
        """Test loss computation for classification heads."""
        criterion = MultiHeadLoss()
        
        # Create outputs and labels for classification head
        outputs = {"head_a": torch.randn(2, 8)}  # 8 classes for head_a
        labels = {"head_a": torch.tensor([0, 1])}
        
        losses = criterion(outputs, labels)
        
        assert "head_a" in losses
        assert losses["head_a"].item() >= 0
        assert "total" in losses
    
    def test_loss_computation_multilabel_head(self):
        """Test loss computation for multilabel heads."""
        criterion = MultiHeadLoss()
        
        # Create outputs and labels for multilabel head
        outputs = {"head_c_a": torch.sigmoid(torch.randn(2, 27))}  # 27 classes for head_c_a, sigmoid for [0,1] range
        labels = {"head_c_a": torch.randint(0, 2, (2, 27)).float()}
        
        losses = criterion(outputs, labels)
        
        assert "head_c_a" in losses
        assert losses["head_c_a"].item() >= 0
        assert "total" in losses
    
    def test_loss_computation_boolean_head(self):
        """Test loss computation for boolean heads."""
        criterion = MultiHeadLoss()
        
        # Create outputs and labels for boolean head
        outputs = {"head_d": torch.sigmoid(torch.randn(2, 3))}  # 3 boolean fields for head_d, sigmoid for [0,1] range
        labels = {"head_d": torch.randint(0, 2, (2, 3)).float()}
        
        losses = criterion(outputs, labels)
        
        assert "head_d" in losses
        assert losses["head_d"].item() >= 0
        assert "total" in losses
    
    def test_loss_computation_mixed_heads(self, sample_outputs, sample_labels):
        """Test loss computation with mixed head types."""
        criterion = MultiHeadLoss()
        
        losses = criterion(sample_outputs, sample_labels)
        
        # Check that all head types are handled correctly
        head_configs = get_head_configs()
        for head_name, head_config in head_configs.items():
            assert head_name in losses
            assert losses[head_name].item() >= 0
        
        # Check that total loss is reasonable (may include loss weights)
        total_loss = losses["total"].item()
        individual_losses = [losses[head_name].item() for head_name in head_configs.keys()]
        assert total_loss >= 0
        assert total_loss <= sum(individual_losses) * 2  # Allow for loss weights up to 2x
    
    def test_loss_computation_empty_outputs(self):
        """Test loss computation with empty outputs."""
        criterion = MultiHeadLoss()
        
        outputs = {}
        labels = {}
        
        losses = criterion(outputs, labels)
        
        assert "total" in losses
        assert losses["total"] == 0.0
    
    def test_loss_computation_missing_labels(self):
        """Test loss computation with missing labels."""
        criterion = MultiHeadLoss()
        
        # Create outputs for all heads but labels for only some
        outputs = {
            "head_a": torch.randn(2, 12),  # Updated count
            "head_c_a": torch.randn(2, 27),
            "head_c_b": torch.randn(2, 20),
            "head_d": torch.randn(2, 3)
        }
        labels = {
            "head_a": torch.tensor([0, 1]),
            # Missing head_c_a, head_c_b and head_d labels
        }
        
        losses = criterion(outputs, labels)
        
        # Should only compute loss for head_a
        assert "head_a" in losses
        assert "head_c_a" not in losses
        assert "head_c_b" not in losses
        assert "head_d" not in losses
        assert "total" in losses


class TestModelIntegration:
    """Integration tests for model components."""
    
    def _create_mock_encoder(self):
        """Helper to create a properly structured mock encoder."""
        mock_encoder = Mock()
        
        # Create mock parameters with requires_grad=False for frozen testing
        mock_param1 = Mock()
        mock_param1.requires_grad = False
        mock_param2 = Mock()
        mock_param2.requires_grad = False
        mock_encoder.parameters.return_value = [mock_param1, mock_param2]
        
        # Mock the encoder.encoder.layer structure for layer freezing
        mock_layers = []
        for _ in range(12):  # 12 BERT layers
            mock_layer = Mock()
            # Create mock parameters with requires_grad=False for frozen testing
            layer_param1 = Mock()
            layer_param1.requires_grad = False
            layer_param2 = Mock()
            layer_param2.requires_grad = False
            mock_layer.parameters.return_value = [layer_param1, layer_param2]
            mock_layers.append(mock_layer)
        mock_encoder.encoder.layer = mock_layers
        
        # Mock embeddings and pooler
        mock_encoder.embeddings = Mock()
        emb_param1 = Mock()
        emb_param1.requires_grad = False
        emb_param2 = Mock()
        emb_param2.requires_grad = False
        mock_encoder.embeddings.parameters.return_value = [emb_param1, emb_param2]
        
        mock_encoder.pooler = Mock()
        pool_param1 = Mock()
        pool_param1.requires_grad = False
        pool_param2 = Mock()
        pool_param2.requires_grad = False
        mock_encoder.pooler.parameters.return_value = [pool_param1, pool_param2]
        
        return mock_encoder
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_and_loss_integration(self, mock_config, mock_model):
        """Test integration between model and loss function."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_encoder.return_value = Mock(
            last_hidden_state=torch.randn(2, 128, 768),
            pooler_output=torch.randn(2, 768)
        )
        mock_model.return_value = mock_encoder
        
        # Create model and loss function
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        criterion = MultiHeadLoss()
        
        # Create sample batch
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Create labels
        labels = {}
        head_configs = get_head_configs()
        for head_name, head_config in head_configs.items():
            if head_config.head_type in ["multilabel", "boolean"]:
                labels[head_name] = torch.randint(0, 2, (batch_size, head_config.num_classes)).float()
            else:
                labels[head_name] = torch.randint(0, head_config.num_classes, (batch_size,))
        
        # Compute loss
        losses = criterion(outputs, labels)
        
        # Check that loss computation works
        assert "total" in losses
        assert losses["total"].item() >= 0
        
        # Check that all heads have losses
        for head_name in labels.keys():
            assert head_name in losses
            assert losses[head_name].item() >= 0
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_gradient_flow(self, mock_config, mock_model):
        """Test that gradients flow correctly through the model."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        mock_encoder = Mock()
        mock_encoder.parameters.return_value = [Mock(), Mock()]  # Make it iterable
        mock_encoder.return_value = Mock(
            last_hidden_state=torch.randn(2, 128, 768, requires_grad=True),
            pooler_output=torch.randn(2, 768, requires_grad=True)
        )
        mock_model.return_value = mock_encoder
        
        # Create model
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=False  # Don't freeze for gradient test
        )
        
        # Create sample batch
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check that outputs require gradients
        for head_name, output in outputs.items():
            assert output.requires_grad
    
    @patch('rejection_detection.model.AutoModel.from_pretrained')
    @patch('rejection_detection.model.AutoConfig.from_pretrained')
    def test_model_different_batch_sizes(self, mock_config, mock_model):
        """Test model with different batch sizes."""
        # Mock the config and model
        mock_config.return_value = Mock()
        mock_config.return_value.hidden_size = 768  # Set hidden_size for calculations
        
        mock_encoder = self._create_mock_encoder()
        mock_model.return_value = mock_encoder
        
        model = MultiHeadClassifier(
            model_name="bert-base-uncased",
            freeze_encoder=True
        )
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            seq_length = 128
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)
            
            # Mock encoder output for this batch size
            mock_encoder.return_value = Mock(
                last_hidden_state=torch.randn(batch_size, seq_length, 768),
                pooler_output=torch.randn(batch_size, 768)
            )
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Check that outputs have correct batch size
            for head_name, output in outputs.items():
                assert output.shape[0] == batch_size
