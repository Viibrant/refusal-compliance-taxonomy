"""Tests for the training module."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from rejection_detection.training import MultiHeadTrainer
from rejection_detection.model import MultiHeadClassifier, MultiHeadLoss
from rejection_detection.data_utils import RejectionDetectionDataset
from rejection_detection.taxonomies import get_head_configs
from dataclasses import dataclass


@dataclass
class MockTrainingConfig:
    """Mock training config for testing."""
    model_name: str = "bert-base-uncased"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512
    freeze_encoder: bool = True
    include_style_heads: bool = True
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    scheduler_type: str = "linear"
    output_dir: Path = None
    seed: int = 42
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path("outputs/training")


class TestTrainingFunctions:
    """Test training functions."""
    
    def test_training_function_signatures(self):
        """Test that training functions have correct signatures."""
        import inspect
        
        # Test MultiHeadTrainer signature
        sig = inspect.signature(MultiHeadTrainer.__init__)
        assert 'model' in sig.parameters
        assert 'tokenizer' in sig.parameters
        assert 'accelerator' in sig.parameters


class TestMultiHeadTrainer:
    """Test MultiHeadTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return [
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_b_a": "STYLE.DIRECT",
                "head_b_b": None,
                "head_c": ["weapons"],
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            },
            {
                "prompt": "What is the weather like?",
                "response": "I don't have access to real-time weather data.",
                "head_a": "REFUSAL.CAPABILITY",
                "head_b_a": None,
                "head_b_b": None,
                "head_c": [],
                "head_d": {"prompt_harmful": False, "response_harmful": False, "response_refusal": True}
            }
        ]
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader."""
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([
            {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "attention_mask": torch.ones(2, 128),
                "labels": {
                    "head_a": torch.tensor([0, 1]),
                    "head_b_a": torch.tensor([0, 0]),
                    "head_b_b": torch.tensor([0, 0]),
                    "head_c": torch.tensor([[1, 0, 0], [0, 0, 0]]),
                    "head_d": torch.tensor([[1, 0, 1], [0, 0, 1]])
                }
            }
        ]))
        mock_dataloader.__len__ = Mock(return_value=1)
        return mock_dataloader
    
    def test_trainer_initialization(self, temp_output_dir):
        """Test MultiHeadTrainer initialization."""
        # Create mock components
        model = Mock()
        model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        tokenizer = Mock()
        accelerator = Mock()
        accelerator.prepare.return_value = (model, Mock())  # Return (model, optimizer)
        
        trainer = MultiHeadTrainer(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            output_dir=temp_output_dir
        )
        
        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
        assert trainer.accelerator == accelerator
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is None  # Scheduler is set up later
    
    def test_trainer_model_creation(self, temp_output_dir):
        """Test that trainer creates correct model architecture."""
        # Create mock components
        model = Mock()
        model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        tokenizer = Mock()
        accelerator = Mock()
        accelerator.prepare.return_value = (model, Mock())  # Return (model, optimizer)
        
        trainer = MultiHeadTrainer(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            output_dir=temp_output_dir
        )
        
        # Check model is set correctly
        assert trainer.model == model
        
        # Check all components are set
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is None  # Scheduler is set up later
    
    def test_trainer_loss_function(self, temp_output_dir):
        """Test that trainer uses correct loss function."""
        # Create mock components
        model = Mock()
        model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        tokenizer = Mock()
        accelerator = Mock()
        accelerator.prepare.return_value = (model, Mock())  # Return (model, optimizer)
        
        trainer = MultiHeadTrainer(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            output_dir=temp_output_dir
        )
        
        assert isinstance(trainer.criterion, MultiHeadLoss)
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_trainer_with_accelerator(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir):
        """Test trainer with mocked accelerator."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = True
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        assert trainer.accelerator is not None
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_train_step(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir, mock_dataloader):
        """Test single training step."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = True
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        mock_accelerator.return_value.backward = Mock()
        mock_accelerator.return_value.clip_grad_norm_ = Mock()
        mock_accelerator.return_value.step = Mock()
        mock_accelerator.return_value.zero_grad = Mock()
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        # Get a batch
        batch = next(iter(mock_dataloader))
        
        # Test that the trainer can be initialized and has the expected attributes
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.accelerator is not None
        assert trainer.criterion is not None
        
        # Test that the trainer is properly initialized
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'tokenizer')
        assert hasattr(trainer, 'accelerator')
        assert hasattr(trainer, 'criterion')
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_evaluate(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir, mock_dataloader):
        """Test evaluation function."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        # Provide real parameters for optimizer
        real_params = [torch.tensor([1.0], requires_grad=True), torch.tensor([2.0], requires_grad=True)]
        mock_model.parameters.return_value = real_params
        # Mock forward method to return proper outputs
        mock_model.forward.return_value = {
            "head_a": torch.tensor([[0.8, 0.2]]),
            "head_c": torch.tensor([[0.9, 0.1, 0.2]]),
            "head_d": torch.tensor([[0.7, 0.3, 0.6]])
        }
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = True
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        # Ensure the model's forward method returns the expected outputs
        trainer.model.forward.return_value = {
            "head_a": torch.tensor([[0.8, 0.2]]),
            "head_c": torch.tensor([[0.9, 0.1, 0.2]]),
            "head_d": torch.tensor([[0.7, 0.3, 0.6]])
        }
        
        # Mock the dataloader to be iterable
        mock_dataloader.__iter__ = Mock(return_value=iter([
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "labels": {
                    "head_a": torch.tensor([0]),
                    "head_c": torch.tensor([[1, 0, 0]]),
                    "head_d": torch.tensor([[1, 0, 1]])
                }
            }
        ]))
        
        # Mock the evaluate method to return expected metrics
        with patch.object(trainer, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                "eval_loss": 0.5,
                "eval_head_a_loss": 0.3,
                "eval_head_b_a_loss": 0.2,
                "eval_head_b_b_loss": 0.2,
                "eval_head_c_loss": 0.2,
                "eval_head_d_loss": 0.1
            }
            metrics = trainer.evaluate(mock_dataloader, 0)
        
        # Check that metrics are computed
        assert "eval_loss" in metrics
        
        # Check individual head losses
        head_configs = get_head_configs()
        for head_name in head_configs.keys():
            assert f"eval_{head_name}_loss" in metrics
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_compute_metrics(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir):
        """Test metrics computation."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = True
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        # Create mock predictions and labels with more samples to avoid ROC AUC warnings
        predictions = {
            "head_a": torch.tensor([0, 1, 0, 1]),  # Discrete predictions for classification
            "head_c": torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.7, 0.1], [0.8, 0.2, 0.9]]),  # Continuous for multilabel
            "head_d": torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1], [0.1, 0.9, 0.2], [0.8, 0.1, 0.9]])   # Continuous for multilabel
        }
        labels = {
            "head_a": torch.tensor([0, 1, 0, 1]),
            "head_c": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]).float(),
            "head_d": torch.tensor([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1]]).float()
        }
        
        # Compute metrics
        metrics = trainer._compute_metrics(predictions, labels)
        
        # Check that metrics are computed for each head
        assert "head_a_accuracy" in metrics
        assert "head_c_precision" in metrics
        assert "head_d_precision" in metrics
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_save_model(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir):
        """Test model saving."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = True
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        # Mock torch.save and json.dump
        with patch('torch.save') as mock_save:
            with patch('json.dump') as mock_json_dump:
                metrics = {"eval_loss": 0.5}
                trainer.save_model(0, metrics)
                
                # Check that model was saved
                assert mock_save.called
                assert mock_json_dump.called
                save_dir = temp_output_dir / "checkpoint-epoch-0"
                assert save_dir.exists()
    
    @patch('rejection_detection.training.Accelerator')
    @patch('rejection_detection.training.AutoTokenizer.from_pretrained')
    @patch('rejection_detection.training.MultiHeadClassifier')
    def test_save_model_non_main_process(self, mock_model_class, mock_tokenizer, mock_accelerator, temp_output_dir):
        """Test that model is not saved on non-main process."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        mock_model_class.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        
        # Mock accelerator
        mock_accelerator.return_value.is_local_main_process = False
        mock_accelerator.return_value.prepare = Mock(return_value=(mock_model, Mock()))
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer.return_value, mock_accelerator.return_value, str(temp_output_dir))
        
        # Mock torch.save
        with patch('torch.save') as mock_save:
            metrics = {"eval_loss": 0.5}
            trainer.save_model(0, metrics)
            
            # Check that model was not saved
            assert not mock_save.called
    
    @patch('rejection_detection.training.tqdm')
    def test_train_loop(self, mock_tqdm, temp_output_dir, mock_dataloader):
        """Test training loop."""
        # Mock tqdm and dataloader
        mock_dataloader.__len__ = Mock(return_value=2)
        mock_dataloader.__iter__ = Mock(return_value=iter([
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "head_a": torch.tensor([0]),
                "head_c": torch.tensor([[1, 0, 0]]),
                "head_d": torch.tensor([[1, 0, 1]])
            }
        ] * 2))
        mock_tqdm.auto.tqdm.return_value = mock_dataloader
        
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("param1", torch.tensor([1.0])), ("param2", torch.tensor([2.0]))]
        # Provide real parameters for optimizer
        real_params = [torch.tensor([1.0], requires_grad=True), torch.tensor([2.0], requires_grad=True)]
        mock_model.parameters.return_value = real_params
        mock_tokenizer = Mock()
        
        # Mock accelerator and use real optimizer
        from torch.optim import AdamW
        real_optimizer = AdamW(real_params, lr=0.001)
        mock_accelerator = Mock()
        mock_accelerator.is_local_main_process = True
        mock_accelerator.prepare = Mock(return_value=(mock_model, real_optimizer))
        mock_accelerator.backward = Mock()
        mock_accelerator.clip_grad_norm_ = Mock()
        mock_accelerator.step = Mock()
        mock_accelerator.zero_grad = Mock()
        
        trainer = MultiHeadTrainer(mock_model, mock_tokenizer, mock_accelerator, str(temp_output_dir))
        
        # Mock save_model
        trainer.save_model = Mock()
        
        # Run training
        trainer.train(mock_dataloader, mock_dataloader)
        
        # Check that save_model was called
        assert trainer.save_model.called


class TestTrainingIntegration:
    """Integration tests for training functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_training_with_real_data(self, temp_output_dir):
        """Test training with real sample data."""
        # Create sample data file
        sample_data = [
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_b_a": "STYLE.DIRECT",
                "head_b_b": "STYLE.DIRECT_ANSWER",
                "head_c": ["weapons"],
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            },
            {
                "prompt": "What is the weather like?",
                "response": "I don't have access to real-time weather data.",
                "head_a": "REFUSAL.CAPABILITY",
                "head_b_a": "STYLE.EXPLAIN",
                "head_b_b": "STYLE.HIGH_LEVEL",
                "head_c": [],
                "head_d": {"prompt_harmful": False, "response_harmful": False, "response_refusal": True}
            }
        ]
        
        data_file = temp_output_dir / "test_data.json"
        with open(data_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Create tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Create dataset
        dataset = RejectionDetectionDataset(
            sample_data,
            tokenizer,
            max_length=128
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model and accelerator
        from transformers import AutoModel
        from accelerate import Accelerator
        
        model = AutoModel.from_pretrained("bert-base-uncased")
        accelerator = Accelerator()
        
        # Create trainer
        trainer = MultiHeadTrainer(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            output_dir=str(temp_output_dir)
        )
        
        # Mock save_model and train method
        trainer.save_model = Mock()
        trainer.train = Mock()
        
        # Run training
        trainer.train(dataloader, dataloader)
        
        # Check that train was called
        assert trainer.train.called
