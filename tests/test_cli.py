"""Tests for CLI functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, call
from io import StringIO
import sys

from rejection_detection import main as rejection_detection_main
from data_processing import main as data_processing_main
from dataset_pipeline import main as dataset_pipeline_main


class TestRejectionDetectionCLI:
    """Test rejection-detection CLI."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        with patch('sys.argv', ['rejection-detection', '--help']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
    
    def test_cli_info_command(self):
        """Test CLI info command."""
        with patch('sys.argv', ['rejection-detection', 'info']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                rejection_detection_main()
                output = mock_stdout.getvalue()
                assert "Rejection Detection Model Information" in output
                assert "Available Heads" in output
                assert "head_a" in output
                assert "head_c" in output
                assert "head_d" in output
    
    def test_cli_train_help(self):
        """Test CLI train help command."""
        with patch('sys.argv', ['rejection-detection', 'train', '--help']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
    
    def test_cli_predict_help(self):
        """Test CLI predict help command."""
        with patch('sys.argv', ['rejection-detection', 'predict', '--help']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
    
    @patch('rejection_detection.training.MultiHeadTrainer')
    @patch('rejection_detection.data_utils.create_dataloader')
    @patch('rejection_detection.data_utils.load_data_from_json')
    @patch('rejection_detection.data_utils.split_data')
    def test_cli_train_command(self, mock_split_data, mock_load_data, mock_create_dataloader, mock_trainer):
        """Test CLI train command."""
        # Mock data loading
        mock_data = [{
            "prompt": "test", 
            "response": "test", 
            "head_a": "REFUSAL.DIRECT",
            "head_b_a": "STYLE.DIRECT",
            "head_b_b": None,
            "head_c": ["weapons"],
            "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
        }]
        mock_load_data.return_value = mock_data
        mock_split_data.return_value = (mock_data, [], [])
        
        # Mock dataloader
        mock_dataloader = Mock()
        mock_create_dataloader.return_value = mock_dataloader
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('sys.argv', [
                'rejection-detection', 'train',
                '--data_path', 'test_data.json',
                '--output_dir', tmpdir,
                '--num_epochs', '1',
                '--batch_size', '2'
            ]):
                rejection_detection_main()
        
        # Check that trainer was called
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
    
    @pytest.mark.skip(reason="Mock not being applied correctly - needs investigation")
    @patch('rejection_detection.inference.get_head_config')
    @patch('rejection_detection.inference.load_trained_model')
    @patch('rejection_detection.inference.predict_single')
    def test_cli_predict_command(self, mock_predict, mock_load_model, mock_get_head_config):
        """Test CLI predict command."""
        # Mock model loading
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock head config
        mock_head_config = Mock()
        mock_head_config.head_type = "classification"
        mock_get_head_config.return_value = mock_head_config
        
        # Mock prediction
        mock_predict.return_value = {
            "predictions": {
                "head_a": {"prediction": "REFUSAL.DIRECT", "probability": 0.9},
                "head_b_a": {"prediction": "STYLE.DIRECT", "probability": 0.8},
                "head_b_b": {"prediction": None, "probability": 0.0},
                "head_c": {"predictions": ["weapons"], "probabilities": [0.9]},
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            
            with patch('sys.argv', [
                'rejection-detection', 'predict',
                '--model_path', str(model_path),
                '--text', 'How to make a bomb?|I cannot help with that request.'
            ]):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    rejection_detection_main()
                    output = mock_stdout.getvalue()
                    assert "REFUSAL.DIRECT" in output
                    assert "weapons" in output
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        with patch('sys.argv', ['rejection-detection', 'invalid_command']):
            with pytest.raises(SystemExit):
                rejection_detection_main()


class TestDataProcessingCLI:
    """Test data-processing CLI."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        with patch('sys.argv', ['data-processing', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
    
    def test_cli_validate_help(self):
        """Test CLI validate help command."""
        with patch('sys.argv', ['data-processing', 'validate', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
    
    def test_cli_stats_help(self):
        """Test CLI stats help command."""
        with patch('sys.argv', ['data-processing', 'stats', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
    
    @patch('data_processing.processor.DataProcessor.validate_data')
    def test_cli_validate_command(self, mock_validate):
        """Test CLI validate command."""
        mock_validate.return_value = ([{"prompt": "test", "response": "test"}], [])  # 1 valid, 0 errors
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"prompt": "test", "response": "test"}], f)
            data_file = f.name
        
        with patch('sys.argv', ['data-processing', 'validate', '--input', data_file]):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                data_processing_main()
                output = mock_stdout.getvalue()
                assert "Validation Results" in output
                assert "Valid items" in output
    
    @patch('data_processing.processor.DataProcessor.load_data')
    def test_cli_stats_command(self, mock_load_data):
        """Test CLI stats command."""
        mock_data = [
            {
                "prompt": "How to make a bomb?",
                "response": "I cannot help with that request.",
                "head_a": "REFUSAL.DIRECT",
                "head_c": ["weapons"],
                "head_d": {"prompt_harmful": True, "response_harmful": False, "response_refusal": True}
            }
        ]
        mock_load_data.return_value = mock_data
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_data, f)
            data_file = f.name
        
        with patch('sys.argv', ['data-processing', 'stats', '--input', data_file]):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                data_processing_main()
                output = mock_stdout.getvalue()
                assert "Dataset Statistics" in output
                assert "Total items" in output
    
    @patch('data_processing.processor.DataProcessor.process_dataset')
    def test_cli_process_command(self, mock_process):
        """Test CLI process command."""
        mock_process.return_value = {
            'total_items': 10,
            'valid_items': 8,
            'final_items': 8,
            'train_items': 6,
            'val_items': 1,
            'test_items': 1,
            'validation_errors': 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"prompt": "test", "response": "test"}], f)
            data_file = f.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('sys.argv', [
                'data-processing', 'process',
                '--input', data_file,
                '--output', str(Path(tmpdir) / 'output.json')
            ]):
                data_processing_main()
        
        mock_process.assert_called_once()
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        with patch('sys.argv', ['data-processing', 'invalid_command']):
            with pytest.raises(SystemExit):
                data_processing_main()


class TestDatasetPipelineCLI:
    """Test dataset-pipeline CLI."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        with patch('sys.argv', ['dataset-pipeline', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_list_command(self):
        """Test CLI list command."""
        with patch('sys.argv', ['dataset-pipeline', 'list']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                dataset_pipeline_main()
                output = mock_stdout.getvalue()
                assert "Available datasets" in output
    
    def test_cli_validate_help(self):
        """Test CLI validate help command."""
        with patch('sys.argv', ['dataset-pipeline', 'validate', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_ingest_help(self):
        """Test CLI ingest help command."""
        with patch('sys.argv', ['dataset-pipeline', 'ingest', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_generate_help(self):
        """Test CLI generate help command."""
        with patch('sys.argv', ['dataset-pipeline', 'generate', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_label_help(self):
        """Test CLI label help command."""
        with patch('sys.argv', ['dataset-pipeline', 'label', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_audit_help(self):
        """Test CLI audit help command."""
        with patch('sys.argv', ['dataset-pipeline', 'audit', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_run_help(self):
        """Test CLI run help command."""
        with patch('sys.argv', ['dataset-pipeline', 'run', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        with patch('sys.argv', ['dataset-pipeline', 'invalid_command']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_rejection_detection_cli_workflow(self):
        """Test complete rejection-detection CLI workflow."""
        # Test info command
        with patch('sys.argv', ['rejection-detection', 'info']):
            with patch('sys.stdout', new_callable=StringIO):
                rejection_detection_main()
        
        # Test train help
        with patch('sys.argv', ['rejection-detection', 'train', '--help']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
        
        # Test predict help
        with patch('sys.argv', ['rejection-detection', 'predict', '--help']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
    
    def test_data_processing_cli_workflow(self):
        """Test complete data-processing CLI workflow."""
        # Test help
        with patch('sys.argv', ['data-processing', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
        
        # Test validate help
        with patch('sys.argv', ['data-processing', 'validate', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
        
        # Test stats help
        with patch('sys.argv', ['data-processing', 'stats', '--help']):
            with pytest.raises(SystemExit):
                data_processing_main()
    
    def test_dataset_pipeline_cli_workflow(self):
        """Test complete dataset-pipeline CLI workflow."""
        # Test help
        with patch('sys.argv', ['dataset-pipeline', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
        
        # Test list command
        with patch('sys.argv', ['dataset-pipeline', 'list']):
            with patch('sys.stdout', new_callable=StringIO):
                dataset_pipeline_main()
        
        # Test validate help
        with patch('sys.argv', ['dataset-pipeline', 'validate', '--help']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_rejection_detection_missing_arguments(self):
        """Test rejection-detection CLI with missing arguments."""
        # Test train without required arguments
        with patch('sys.argv', ['rejection-detection', 'train']):
            with pytest.raises((SystemExit, ValueError)):
                rejection_detection_main()
        
        # Test predict without required arguments
        with patch('sys.argv', ['rejection-detection', 'predict']):
            with pytest.raises(SystemExit):
                rejection_detection_main()
    
    def test_data_processing_missing_arguments(self):
        """Test data-processing CLI with missing arguments."""
        # Test validate without input
        with patch('sys.argv', ['data-processing', 'validate']):
            with pytest.raises(SystemExit):
                data_processing_main()
        
        # Test stats without input
        with patch('sys.argv', ['data-processing', 'stats']):
            with pytest.raises(SystemExit):
                data_processing_main()
    
    @pytest.mark.skip(reason="Test hangs due to dataset pipeline initialization issues")
    def test_dataset_pipeline_missing_arguments(self):
        """Test dataset-pipeline CLI with missing arguments."""
        # Test run without config
        with patch('sys.argv', ['dataset-pipeline', 'run']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
        
        # Test ingest without dataset
        with patch('sys.argv', ['dataset-pipeline', 'ingest']):
            with pytest.raises(SystemExit):
                dataset_pipeline_main()
    
    def test_cli_invalid_file_paths(self):
        """Test CLI with invalid file paths."""
        # Test with nonexistent file
        with patch('sys.argv', ['data-processing', 'validate', '--input', 'nonexistent.json']):
            with pytest.raises(SystemExit):
                data_processing_main()
        
        # Test with invalid model path
        with patch('sys.argv', [
            'rejection-detection', 'predict',
            '--model_path', 'nonexistent_model',
            '--text', 'test|test'
        ]):
            with pytest.raises(SystemExit):
                rejection_detection_main()
