"""Rejection detection package."""

__version__ = "0.1.0"

# Import main components
from .model import MultiHeadClassifier, MultiHeadLoss
from .training import MultiHeadTrainer, create_trainer
from .data_utils import (
    RejectionDetectionDataset,
    create_dataloader,
    load_data_from_json,
    load_data_from_csv,
    create_sample_data,
)
from .taxonomies import (
    get_head_configs,
    get_head_config,
    get_all_head_configs,
    get_label_to_id_mapping,
    get_id_to_label_mapping,
    OutcomeType,
    RefusalStyle,
    ComplianceStyle,
    HarmCategory,
)
from .inference import load_trained_model, predict_single, predict_batch

# Legacy simple detector for backward compatibility
class RejectionDetector:
    """Main class for detecting rejections."""
    
    def __init__(self) -> None:
        """Initialize the rejection detector."""
        pass
    
    def detect(self, data: str) -> bool:
        """
        Detect if the given data contains a rejection.
        
        Args:
            data: The data to analyze for rejections
            
        Returns:
            True if a rejection is detected, False otherwise
        """
        import re
        
        # Placeholder implementation - look for explicit rejection words with word boundaries
        rejection_words = ["reject", "deny", "refuse", "decline"]
        data_lower = data.lower()
        
        for word in rejection_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, data_lower):
                return True
        return False


def main() -> None:
    """Main entry point for the rejection-detection CLI."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Rejection Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_path", type=str, help="Path to training data")
    train_parser.add_argument("--data_format", type=str, choices=["json", "csv", "huggingface"], default="json", help="Data format")
    train_parser.add_argument("--use_sample_data", action="store_true", help="Use sample data")
    train_parser.add_argument("--sample_size", type=int, default=1000, help="Number of sample data points")
    train_parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name")
    train_parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights")
    train_parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    train_parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    train_parser.add_argument("--scheduler_type", type=str, choices=["linear", "cosine"], default="linear", help="Scheduler type")
    train_parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    train_parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    train_parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    train_parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations")
    train_parser.add_argument("--save_steps", type=int, default=1000, help="Steps between saves")
    train_parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging")
    train_parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    train_parser.add_argument("--loss_weights", type=str, help="Loss weights JSON")
    train_parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--include_style_heads", action="store_true", default=True, help="Include style heads")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--text", type=str, help="Text to analyze (format: 'prompt|response')")
    predict_parser.add_argument("--input_file", type=str, help="Input file with texts")
    predict_parser.add_argument("--output_file", type=str, help="Output file for results")
    predict_parser.add_argument("--return_probabilities", action="store_true", help="Return full probability distributions")
    predict_parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from .train import main as train_main
        # Convert args to sys.argv for train_main
        train_args = []
        for k, v in vars(args).items():
            if k != "command" and v is not None:
                if isinstance(v, bool):
                    if v:
                        train_args.append(f"--{k}")
                else:
                    train_args.append(f"--{k}={v}")
        sys.argv = ["train.py"] + train_args
        train_main()
    elif args.command == "predict":
        from .inference import main as inference_main
        # Convert args to sys.argv for inference_main
        predict_args = []
        for k, v in vars(args).items():
            if k != "command" and v is not None:
                if isinstance(v, bool):
                    if v:
                        predict_args.append(f"--{k}")
                else:
                    predict_args.append(f"--{k}={v}")
        sys.argv = ["inference.py"] + predict_args
        inference_main()
    elif args.command == "info":
        print("Rejection Detection Model Information:")
        print(f"Version: {__version__}")
        print("\nAvailable Heads:")
        for head_name, config in get_all_head_configs().items():
            print(f"  {head_name}: {config.num_classes} classes ({config.head_type})")
            if config.class_names:
                print(f"    Labels: {config.class_names}")
    else:
        parser.print_help()
