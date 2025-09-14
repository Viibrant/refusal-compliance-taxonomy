"""Multi-head encoder-only model for rejection detection."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoConfig, AutoTokenizer
from .taxonomies import get_head_configs, get_head_config


class MultiHeadClassifier(nn.Module):
    """Multi-head classifier for rejection detection."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dropout_prob: float = 0.1,
        head_dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        head_hidden_size: Optional[int] = None,
    ):
        """
        Initialize the multi-head classifier.
        
        Args:
            model_name: Name of the pre-trained model to use
            hidden_dropout_prob: Dropout probability for hidden layers
            head_dropout_prob: Dropout probability for classification heads
            freeze_encoder: Whether to freeze most encoder weights (keeps last layer trainable)
            head_hidden_size: Hidden size for classification heads (defaults to encoder hidden size)
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            # Freeze all layers except the last one
            for layer in self.encoder.encoder.layer[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
            # Also freeze embeddings and pooler
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for param in self.encoder.pooler.parameters():
                param.requires_grad = False
            # Last encoder layer remains trainable
        
        # Set hidden size for heads
        self.hidden_size = self.config.hidden_size
        self.head_hidden_size = head_hidden_size or self.hidden_size
        
        # Dropout layers
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        self.head_dropout = nn.Dropout(head_dropout_prob)
        
        # Projection layer to head hidden size if needed
        if self.head_hidden_size != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.head_hidden_size)
        else:
            self.projection = nn.Identity()
        
        # Initialize classification heads
        self.heads = nn.ModuleDict()
        self._initialize_heads()
        
    def _initialize_heads(self):
        """Initialize all classification heads."""
        for head_name, head_config in get_head_configs().items():
            self.heads[head_name] = self._create_head(
                head_config.num_classes,
                head_config.head_type
            )
    
    def _create_head(self, num_classes: int, head_type: str) -> nn.Module:
        """Create a classification head."""
        if head_type in ["multilabel", "boolean"]:
            # Multi-label/Boolean classification (sigmoid activation)
            return nn.Sequential(
                nn.Linear(self.head_hidden_size, self.head_hidden_size // 2),
                nn.ReLU(),
                self.head_dropout,
                nn.Linear(self.head_hidden_size // 2, num_classes),
                nn.Sigmoid()
            )
        else:
            # Single-label classification (no activation, use CrossEntropyLoss)
            return nn.Sequential(
                nn.Linear(self.head_hidden_size, self.head_hidden_size // 2),
                nn.ReLU(),
                self.head_dropout,
                nn.Linear(self.head_hidden_size // 2, num_classes)
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        head_a_labels: Optional[torch.Tensor] = None,
        head_b_a_labels: Optional[torch.Tensor] = None,
        head_b_b_labels: Optional[torch.Tensor] = None,
        head_c_labels: Optional[torch.Tensor] = None,
        head_d_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT)
            head_a_labels: Labels for head A (optional, for training)
            head_b_a_labels: Labels for head B.A (optional, for training)
            head_b_b_labels: Labels for head B.B (optional, for training)
            head_c_labels: Labels for head C (optional, for training)
            head_d_labels: Labels for head D (optional, for training)
            
        Returns:
            Dictionary containing logits for each head
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = encoder_outputs.pooler_output
        if pooled_output is None:
            # Fallback to mean pooling if no pooler
            pooled_output = self._mean_pooling(encoder_outputs.last_hidden_state, attention_mask)
        
        # Apply dropout and projection
        hidden_states = self.hidden_dropout(pooled_output)
        hidden_states = self.projection(hidden_states)
        
        # Get outputs from all heads
        outputs = {}
        for head_name in self.heads.keys():
            outputs[head_name] = self.heads[head_name](hidden_states)
        
        return outputs
    
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling of hidden states."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_head_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        head_a_prediction: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get outputs from all heads, with conditional logic for style heads.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            head_a_prediction: Predicted labels from head A (for conditional heads)
            
        Returns:
            Dictionary containing outputs for each head
        """
        # Get base outputs
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        
        # Apply conditional logic for style heads
        if head_a_prediction is not None:
            # Determine which style head to use based on head A prediction
            batch_size = head_a_prediction.size(0)
            
            # Create masks for refusal and compliance
            refusal_mask = self._is_refusal_prediction(head_a_prediction)
            compliance_mask = self._is_compliance_prediction(head_a_prediction)
            
            # Zero out inappropriate style head outputs
            if "head_b_a" in outputs:
                outputs["head_b_a"] = outputs["head_b_a"] * refusal_mask.unsqueeze(-1)
            if "head_b_b" in outputs:
                outputs["head_b_b"] = outputs["head_b_b"] * compliance_mask.unsqueeze(-1)
        
        return outputs
    
    def _is_refusal_prediction(self, predictions: torch.Tensor) -> torch.Tensor:
        """Check if predictions are refusal types."""
        # Get refusal indices dynamically from taxonomy
        from .taxonomies import OutcomeType
        refusal_types = [OutcomeType.REFUSAL_DIRECT, OutcomeType.REFUSAL_SOFT, 
                        OutcomeType.REFUSAL_PARTIAL, OutcomeType.REFUSAL_CAPABILITY, 
                        OutcomeType.REFUSAL_OVER]
        refusal_indices = [list(OutcomeType).index(t) for t in refusal_types]
        return torch.isin(predictions, torch.tensor(refusal_indices, device=predictions.device))
    
    def _is_compliance_prediction(self, predictions: torch.Tensor) -> torch.Tensor:
        """Check if predictions are compliance types."""
        # Get compliance indices dynamically from taxonomy
        from .taxonomies import OutcomeType
        compliance_types = [OutcomeType.COMPLY_BENIGN, OutcomeType.COMPLY_UNSAFE, 
                           OutcomeType.COMPLY_TRANSFORM, OutcomeType.COMPLY_CONDITIONAL,
                           OutcomeType.COMPLY_EDUCATIONAL, OutcomeType.COMPLY_REDIRECTED,
                           OutcomeType.COMPLY_PARTIAL_SAFE]
        compliance_indices = [list(OutcomeType).index(t) for t in compliance_types]
        return torch.isin(predictions, torch.tensor(compliance_indices, device=predictions.device))
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_probabilities: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            return_probabilities: Whether to return probabilities or just predictions
            
        Returns:
            Dictionary containing predictions for each head
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            
            predictions = {}
            for head_name, logits in outputs.items():
                head_config = get_head_config(head_name)
                
                if head_config.head_type == "multilabel":
                    # For multilabel, use threshold
                    probs = logits
                    preds = (probs > 0.5).float()
                    if return_probabilities:
                        predictions[head_name] = probs
                    else:
                        predictions[head_name] = preds
                else:
                    # For single-label, use argmax
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    if return_probabilities:
                        predictions[head_name] = probs
                    else:
                        predictions[head_name] = preds
            
            return predictions


class MultiHeadLoss(nn.Module):
    """Loss function for multi-head classification."""
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-head loss.
        
        Args:
            loss_weights: Optional weights for each head's loss
        """
        super().__init__()
        
        self.loss_weights = loss_weights or {
            "head_a": 1.0,
            "head_b_a": 0.5,
            "head_b_b": 0.5,
            "head_c": 1.0,
            "head_d": 1.0,
        }
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        head_a_predictions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for all heads.
        
        Args:
            outputs: Model outputs for each head
            labels: Ground truth labels for each head
            head_a_predictions: Predictions from head A (for conditional losses)
            
        Returns:
            Dictionary containing losses for each head
        """
        losses = {}
        total_loss = 0.0
        
        for head_name, logits in outputs.items():
            if head_name not in labels:
                continue
                
            head_config = get_head_config(head_name)
            head_labels = labels[head_name]
            
            if head_config.head_type in ["multilabel", "boolean"]:
                # Multi-label/Boolean binary cross-entropy
                loss = self.bce_loss(logits, head_labels.float())
            else:
                # Single-label cross-entropy
                loss = self.ce_loss(logits, head_labels)
            
            # Apply conditional logic for style heads
            if head_name in ["head_b_a", "head_b_b"] and head_a_predictions is not None:
                # Only compute loss for appropriate cases
                if head_name == "head_b_a":
                    # Only for refusal cases
                    mask = self._is_refusal_prediction(head_a_predictions)
                else:  # head_b_b
                    # Only for compliance cases
                    mask = self._is_compliance_prediction(head_a_predictions)
                
                if mask.sum() > 0:
                    loss = loss * mask.float().mean()
                else:
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            losses[head_name] = loss
            total_loss += self.loss_weights.get(head_name, 1.0) * loss
        
        losses["total"] = total_loss
        return losses
    
    def _is_refusal_prediction(self, predictions: torch.Tensor) -> torch.Tensor:
        """Check if predictions are refusal types."""
        # Get refusal indices dynamically from taxonomy
        from .taxonomies import OutcomeType
        refusal_types = [OutcomeType.REFUSAL_DIRECT, OutcomeType.REFUSAL_SOFT, 
                        OutcomeType.REFUSAL_PARTIAL, OutcomeType.REFUSAL_CAPABILITY, 
                        OutcomeType.REFUSAL_OVER]
        refusal_indices = [list(OutcomeType).index(t) for t in refusal_types]
        return torch.isin(predictions, torch.tensor(refusal_indices, device=predictions.device))
    
    def _is_compliance_prediction(self, predictions: torch.Tensor) -> torch.Tensor:
        """Check if predictions are compliance types."""
        # Get compliance indices dynamically from taxonomy
        from .taxonomies import OutcomeType
        compliance_types = [OutcomeType.COMPLY_BENIGN, OutcomeType.COMPLY_UNSAFE, 
                           OutcomeType.COMPLY_TRANSFORM, OutcomeType.COMPLY_CONDITIONAL,
                           OutcomeType.COMPLY_EDUCATIONAL, OutcomeType.COMPLY_REDIRECTED,
                           OutcomeType.COMPLY_PARTIAL_SAFE]
        compliance_indices = [list(OutcomeType).index(t) for t in compliance_types]
        return torch.isin(predictions, torch.tensor(compliance_indices, device=predictions.device))
