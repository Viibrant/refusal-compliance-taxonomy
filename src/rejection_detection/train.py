"""Main training script for multi-head rejection detection model."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer

from .training import create_trainer
from .data_utils import (
    load_data_from_json,
    load_data_from_csv,
    load_data_from_huggingface,
    create_dataloader,
    split_data,
    create_sample_data,
    save_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-head rejection detection model")
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to training data (JSON, CSV, or Hugging Face dataset name)"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["json", "csv", "huggingface"],
        default="json",
        help="Format of the input data"
    )
    parser.add_argument(
        "--use_sample_data",
        action="store_true",
        help="Use sample data for testing"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of sample data points to generate"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained model name"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder weights"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["linear", "cosine"],
        default="linear",
        help="Type of learning rate scheduler"
    )
    
    # Data split arguments
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio for training set"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio for validation set"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio for test set"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Steps between evaluations"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Steps between model saves"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Steps between logging"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Patience for early stopping"
    )
    
    # Loss weights
    parser.add_argument(
        "--loss_weights",
        type=str,
        help="JSON string of loss weights for each head"
    )
    
    # Other arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--include_style_heads",
        action="store_true",
        default=True,
        help="Include style heads (B.A and B.B)"
    )
    
    return parser.parse_args()


def load_training_data(args) -> tuple:
    """Load training data based on arguments."""
    if args.use_sample_data:
        logger.info(f"Generating {args.sample_size} sample data points")
        data = create_sample_data(args.sample_size)
    else:
        if not args.data_path:
            raise ValueError("data_path must be provided when not using sample data")
        
        logger.info(f"Loading data from {args.data_path}")
        
        if args.data_format == "json":
            data = load_data_from_json(args.data_path)
        elif args.data_format == "csv":
            data = load_data_from_csv(args.data_path)
        elif args.data_format == "huggingface":
            data = load_data_from_huggingface(args.data_path)
        else:
            raise ValueError(f"Unknown data format: {args.data_format}")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    
    return train_data, val_data, test_data


def setup_loss_weights(args) -> Optional[Dict[str, float]]:
    """Set up loss weights from arguments."""
    if args.loss_weights:
        try:
            return json.loads(args.loss_weights)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid loss_weights JSON: {e}")
            raise
    return None


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training arguments
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    train_data, val_data, test_data = load_training_data(args)
    
    # Save data splits for reference
    save_data(train_data, output_dir / "train_data.json")
    save_data(val_data, output_dir / "val_data.json")
    save_data(test_data, output_dir / "test_data.json")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    train_dataloader = create_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_dataloader = create_dataloader(
        data=val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Set up loss weights
    loss_weights = setup_loss_weights(args)
    
    # Create trainer
    trainer, tokenizer = create_trainer(
        model_name=args.model_name,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        scheduler_type=args.scheduler_type,
        loss_weights=loss_weights,
        freeze_encoder=args.freeze_encoder,
    )
    
    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Train samples: {len(train_data)}")
    logger.info(f"  Val samples: {len(val_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    logger.info(f"  Include style heads: {args.include_style_heads}")
    
    # Start training
    logger.info("Starting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
