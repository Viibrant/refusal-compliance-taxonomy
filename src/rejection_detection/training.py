"""Training loop for multi-head rejection detection model."""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .model import MultiHeadClassifier, MultiHeadLoss
from .taxonomies import get_head_configs, get_head_config, HeadConfig


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadTrainer:
    """Trainer for multi-head rejection detection model."""
    
    def __init__(
        self,
        model: MultiHeadClassifier,
        tokenizer: AutoTokenizer,
        accelerator: Accelerator,
        output_dir: str = "./outputs",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "linear",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The multi-head model to train
            tokenizer: Tokenizer for the model
            accelerator: Accelerate accelerator for distributed training
            output_dir: Directory to save outputs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            scheduler_type: Type of learning rate scheduler
            loss_weights: Weights for each head's loss
        """
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        self.criterion = MultiHeadLoss(loss_weights)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer(learning_rate, weight_decay)
        
        # Prepare model and optimizer with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # Initialize scheduler (will be set up after getting total steps)
        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        # Training state
        self.global_step = 0
        self.best_metrics = {}
        
    def _setup_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
        """Set up the optimizer."""
        # Separate parameters for encoder and heads
        encoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)
        
        # Use different learning rates for encoder and heads
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": learning_rate},
            {"params": head_params, "lr": learning_rate * 2},  # Higher LR for heads
        ], weight_decay=weight_decay)
        
        return optimizer
    
    def setup_scheduler(self, num_training_steps: int):
        """Set up the learning rate scheduler."""
        if self.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def train_epoch(
        self,
        train_dataloader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        head_losses = {head_name: 0.0 for head_name in get_head_configs().keys()}
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            
            # Compute loss
            losses = self.criterion(outputs, batch["labels"])
            
            # Backward pass
            self.accelerator.backward(losses["total"])
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += losses["total"].item()
            for head_name, loss in losses.items():
                if head_name != "total":
                    head_losses[head_name] += loss.item()
            
            self.global_step += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "lr": f"{current_lr:.2e}",
                })
                
                if self.accelerator.is_local_main_process:
                    logger.info(
                        f"Epoch {epoch}, Step {self.global_step}, "
                        f"Loss: {losses['total'].item():.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
        
        # Average losses
        avg_total_loss = total_loss / num_batches
        avg_head_losses = {head_name: loss / num_batches for head_name, loss in head_losses.items()}
        
        return {
            "train_loss": avg_total_loss,
            **{f"train_{head_name}_loss": loss for head_name, loss in avg_head_losses.items()}
        }
    
    def evaluate(
        self,
        eval_dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        head_losses = {head_name: 0.0 for head_name in get_head_configs().keys()}
        
        # Collect predictions and labels for metrics
        all_predictions = {head_name: [] for head_name in get_head_configs().keys()}
        all_labels = {head_name: [] for head_name in get_head_configs().keys()}
        
        with torch.no_grad():
            for batch in tqdm(
                eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                
                # Compute loss
                losses = self.criterion(outputs, batch["labels"])
                
                # Update loss metrics
                total_loss += losses["total"].item()
                for head_name, loss in losses.items():
                    if head_name != "total":
                        head_losses[head_name] += loss.item()
                
                # Collect predictions and labels
                for head_name, logits in outputs.items():
                    if head_name not in batch["labels"]:
                        continue  # Skip if label is missing
                        
                    head_config = get_head_config(head_name)
                    labels = batch["labels"][head_name]
                    
                    if head_config.head_type in ["multilabel", "boolean"]:
                        # Multi-label/Boolean: use threshold
                        predictions = (torch.sigmoid(logits) > 0.5).float()
                    else:
                        # Single-label: use argmax
                        predictions = torch.argmax(logits, dim=-1)
                    
                    # Ensure correct data types
                    if head_config.head_type in ["multilabel", "boolean"]:
                        all_predictions[head_name].append(predictions.cpu().float())
                        all_labels[head_name].append(labels.cpu().float())
                    else:
                        all_predictions[head_name].append(predictions.cpu().long())
                        all_labels[head_name].append(labels.cpu().long())
        
        # Concatenate all predictions and labels
        for head_name in get_head_configs().keys():
            if all_predictions[head_name]:  # Only concatenate if not empty
                all_predictions[head_name] = torch.cat(all_predictions[head_name], dim=0)
                all_labels[head_name] = torch.cat(all_labels[head_name], dim=0)
            else:
                # Remove empty entries
                del all_predictions[head_name]
                del all_labels[head_name]
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_labels)
        
        # Add loss metrics
        num_batches = len(eval_dataloader)
        metrics["eval_loss"] = total_loss / num_batches
        for head_name, loss in head_losses.items():
            metrics[f"eval_{head_name}_loss"] = loss / num_batches
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute evaluation metrics for all heads."""
        metrics = {}
        
        for head_name, preds in predictions.items():
            head_config = get_head_config(head_name)
            true_labels = labels[head_name]
            
            if head_config.head_type in ["multilabel", "boolean"]:
                # Multi-label metrics
                # Convert to numpy for sklearn
                preds_np = preds.numpy()
                labels_np = true_labels.numpy()
                
                # Convert predictions to binary (threshold at 0.5)
                preds_binary = (preds_np > 0.5).astype(int)
                
                # Compute per-label metrics
                try:
                    # Check if we have sufficient class diversity
                    if labels_np.shape[1] > 1:  # Multiple labels
                        # Check each label column for class diversity
                        valid_labels = []
                        for i in range(labels_np.shape[1]):
                            unique_classes = np.unique(labels_np[:, i])
                            if len(unique_classes) > 1:  # More than one class
                                valid_labels.append(i)
                        
                        if valid_labels:
                            # Only compute metrics for labels with class diversity
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                labels_np[:, valid_labels], 
                                preds_binary[:, valid_labels], 
                                average="macro", 
                                zero_division=0
                            )
                        else:
                            # All labels have only one class, compute simple accuracy
                            accuracy = np.mean(labels_np == preds_binary)
                            precision = recall = f1 = accuracy
                    else:
                        # Single label case
                        unique_classes = np.unique(labels_np)
                        if len(unique_classes) > 1:
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                labels_np, preds_binary, average="macro", zero_division=0
                            )
                        else:
                            # Only one class, compute simple accuracy
                            accuracy = np.mean(labels_np == preds_binary)
                            precision = recall = f1 = accuracy
                except ValueError as e:
                    logger.debug(f"Metrics calculation skipped for {head_name}: {e}")
                    precision = recall = f1 = 0.0
                
                # Compute AUC (average across labels)
                try:
                    # Check if we have sufficient class diversity for AUC calculation
                    if labels_np.shape[1] > 1:  # Multiple labels
                        # Check each label column for class diversity
                        valid_labels = []
                        for i in range(labels_np.shape[1]):
                            unique_classes = np.unique(labels_np[:, i])
                            if len(unique_classes) > 1:  # More than one class
                                valid_labels.append(i)
                        
                        if valid_labels:
                            # Only compute AUC for labels with class diversity
                            auc = roc_auc_score(
                                labels_np[:, valid_labels], 
                                preds_np[:, valid_labels], 
                                average="macro"
                            )
                        else:
                            auc = 0.0
                    else:
                        # Single label case
                        unique_classes = np.unique(labels_np)
                        if len(unique_classes) > 1:
                            auc = roc_auc_score(labels_np, preds_np)
                        else:
                            auc = 0.0
                except (ValueError, IndexError) as e:
                    logger.debug(f"AUC calculation skipped for {head_name}: insufficient class diversity")
                    auc = 0.0
                
                metrics[f"{head_name}_precision"] = precision
                metrics[f"{head_name}_recall"] = recall
                metrics[f"{head_name}_f1"] = f1
                metrics[f"{head_name}_auc"] = auc
                
            else:
                # Single-label metrics
                preds_np = preds.numpy()
                labels_np = true_labels.numpy()
                
                # Handle case where we have only one class
                if len(np.unique(labels_np)) == 1:
                    # All labels are the same class
                    accuracy = 1.0 if np.all(preds_np == labels_np) else 0.0
                    precision = recall = f1 = 1.0 if accuracy == 1.0 else 0.0
                else:
                    accuracy = accuracy_score(labels_np, preds_np)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels_np, preds_np, average="weighted", zero_division=0
                    )
                
                metrics[f"{head_name}_accuracy"] = accuracy
                metrics[f"{head_name}_precision"] = precision
                metrics[f"{head_name}_recall"] = recall
                metrics[f"{head_name}_f1"] = f1
        
        return metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, float]):
        """Save the model and tokenizer."""
        if not self.accelerator.is_local_main_process:
            return
        
        save_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        save_dir.mkdir(exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), save_dir / "pytorch_model.bin")
        
        # Save model config
        model_config = {
            "model_name": unwrapped_model.model_name,
            "hidden_size": unwrapped_model.hidden_size,
            "head_hidden_size": unwrapped_model.head_hidden_size,
            "freeze_encoder": unwrapped_model.freeze_encoder,
        }
        with open(save_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training arguments and metrics
        training_info = {
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "model_name": self.model.model_name,
        }
        
        with open(save_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        eval_steps: int = 500,
        save_steps: int = 1000,
        log_interval: int = 100,
        early_stopping_patience: int = 3,
    ):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            num_epochs: Number of training epochs
            eval_steps: Steps between evaluations
            save_steps: Steps between model saves
            log_interval: Steps between logging
            early_stopping_patience: Patience for early stopping
        """
        # Prepare dataloaders with accelerator
        train_dataloader = self.accelerator.prepare(train_dataloader)
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Set up scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        self.setup_scheduler(num_training_steps)
        
        # Training loop
        best_eval_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_dataloader, epoch, log_interval)
            
            # Evaluate if eval_dataloader is provided
            if eval_dataloader is not None and epoch == 0:  # Only evaluate on first epoch for now
                try:
                    eval_metrics = self.evaluate(eval_dataloader, epoch)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}. Skipping evaluation.")
                    eval_metrics = {}
                
                # Combine metrics
                all_metrics = {**train_metrics, **eval_metrics}
                
                # Check for improvement if we have eval metrics
                if eval_metrics and "eval_loss" in eval_metrics:
                    current_eval_loss = eval_metrics["eval_loss"]
                    if current_eval_loss < best_eval_loss:
                        best_eval_loss = current_eval_loss
                        patience_counter = 0
                        self.best_metrics = all_metrics
                        
                        # Save best model
                        self.save_model(epoch, all_metrics)
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    # No eval metrics, just save periodically
                    if epoch % save_steps == 0:
                        self.save_model(epoch, train_metrics)
                
                # Log metrics
                if self.accelerator.is_local_main_process:
                    logger.info(f"Epoch {epoch} Metrics:")
                    for key, value in all_metrics.items():
                        logger.info(f"  {key}: {value:.4f}")
            else:
                # No evaluation, just save periodically
                if epoch % save_steps == 0:
                    self.save_model(epoch, train_metrics)
        
        # Save final model
        if self.accelerator.is_local_main_process:
            final_metrics = self.best_metrics if self.best_metrics else train_metrics
            self.save_model(epoch, final_metrics)
            logger.info("Training completed!")


def create_trainer(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./outputs",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    max_grad_norm: float = 1.0,
    scheduler_type: str = "linear",
    loss_weights: Optional[Dict[str, float]] = None,
    freeze_encoder: bool = False,
) -> Tuple[MultiHeadTrainer, AutoTokenizer]:
    """
    Create a trainer and tokenizer.
    
    Args:
        model_name: Name of the pre-trained model
        output_dir: Directory to save outputs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm
        scheduler_type: Type of scheduler
        loss_weights: Loss weights for each head
        freeze_encoder: Whether to freeze encoder
        
    Returns:
        Tuple of (trainer, tokenizer)
    """
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize model
    model = MultiHeadClassifier(
        model_name=model_name,
        freeze_encoder=freeze_encoder,
    )
    
    # Initialize trainer
    trainer = MultiHeadTrainer(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        scheduler_type=scheduler_type,
        loss_weights=loss_weights,
    )
    
    return trainer, tokenizer
