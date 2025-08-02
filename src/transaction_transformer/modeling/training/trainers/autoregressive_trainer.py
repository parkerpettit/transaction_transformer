"""
Autoregressive training trainer for transaction transformer.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..base.base_trainer import BaseTrainer
import wandb
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig


class AutoregressiveTrainer(BaseTrainer):
    """Trainer for autoregressive training (next token prediction)."""
    
    def __init__(
        self,
        model: nn.Module,
        schema: FieldSchema,
        config: ModelConfig,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        wandb_run: Optional[Any] = None
    ):
        super().__init__(model, schema, config, device, train_loader, val_loader, optimizer, scheduler)
        self.wandb_run = wandb_run
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        for batch in self.train_bar:
            # Move batch to device
            cat_in = batch["cat"].to(self.device) # (B, L-1, C)
            cont_in = batch["cont"].to(self.device) # (B, L-1, F)
            labels_cat = batch["labels_cat"].to(self.device) # (B, C)
            labels_cont = batch["labels_cont"].to(self.device) # (B, F)
            
            # Forward pass with causal masking
            logits = self.model(
                cat=cat_in,
                cont=cont_in,
                causal=True  # AR uses causal attention
            )
            
            # Compute loss
            loss = self.compute_loss(logits, labels_cat, labels_cont)
        
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss
            num_batches += 1
            
            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        return {"loss": total_loss.item() / num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_bar:
                # Move batch to device

                cat_in = batch["cat"].to(self.device)
                cont_in = batch["cont"].to(self.device)
                labels_cat = batch["labels_cat"].to(self.device)
                labels_cont = batch["labels_cont"].to(self.device)
                
                # Forward pass with causal masking
                logits = self.model(
                    cat=cat_in,
                    cont=cont_in,
                    causal=True  # AR uses causal attention
                )
                
                # Compute loss
                loss = self.compute_loss(logits, labels_cat, labels_cont)
                
                total_loss += loss
                num_batches += 1
        
        return {"val_loss": total_loss.item() / num_batches}
    
    def compute_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
        """Compute loss for autoregressive training using unified loss. 
        labels_cat and labels_cont are the ground truth labels for the categorical and continuous features, respectively, of shape (B, C) and (B, F).
        logits is a dictionary of length K = C + F, where each key is a feature name and each value is a (B, L, V_feature) tensor of logits. 
        For AR training, we want the model to predict the next transaction, so we use the last position of the output.
        Returns the total loss.

        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # For AR training, we want to predict the next transaction given the context
        # The model outputs logits for the next transaction at the last position
        
        # Handle categorical features
        for i, feature_name in enumerate(self.schema.cat_features):
            feature_logits = logits[feature_name][:, -1, :]  # (B, vocab_size) - predict next transaction
            feature_targets = labels_cat[:, i]  # (B,) - target id for this feature
            feature_targets = self.label_smoothing(feature_targets, self.schema.cat_encoders[feature_name].vocab_size)
            total_loss = total_loss + self.loss_fn(feature_logits, feature_targets)
            
        # Handle continuous features  
        for i, feature_name in enumerate(self.schema.cont_features):
            feature_logits = logits[feature_name][:, -1, :]  # (B, num_bins) - predict next transaction
            feature_targets = labels_cont[:, i]  # (B,) - binned target id for this feature
            feature_targets = self.neighbor_label_smoothing(feature_targets, self.schema.cont_binners[feature_name].num_bins)
            total_loss = total_loss + self.loss_fn(feature_logits, feature_targets)
        
        return total_loss



