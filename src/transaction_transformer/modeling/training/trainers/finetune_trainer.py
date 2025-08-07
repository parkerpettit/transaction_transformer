"""
Finetuning trainer for transaction transformer.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..base.base_trainer import BaseTrainer
from transaction_transformer.config.config import Config
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.modeling.training.utils.data_utils import calculate_positive_weight_from_labels

class FinetuneTrainer(BaseTrainer):
    """Trainer for finetuning models on specific tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        schema: FieldSchema,
        config: Config,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        super().__init__(model, schema, config, device, train_loader, val_loader, optimizer, scheduler)
        
        # Calculate positive weight from dataset if using default value
        pos_weight = config.model.training.positive_weight
        if pos_weight == 1.0:  # Default value, calculate from data
            print("Calculating positive weight from dataset...")
            # Get all labels from training dataset
            all_labels = []
            for batch in train_loader:
                all_labels.append(batch["downstream_label"])
            all_labels = torch.cat(all_labels)
            pos_weight = calculate_positive_weight_from_labels(all_labels)
            print(f"Using calculated positive weight: {pos_weight:.2f}")
        
        # Use positive weight to handle class imbalance
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        print(f"Using positive weight: {pos_weight_tensor}")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.val_loss_fn = nn.BCEWithLogitsLoss() # no pos_weight for validation

    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch. Returns logits and labels, both of shape (B,)"""
        cat_in = batch["cat"].to(self.device) #  (B, L, C)
        cont_in = batch["cont"].to(self.device) # (B, L, F)
        labels = batch["downstream_label"].to(self.device) # (B,)
        logits = self.model(
            cat=cat_in,
            cont=cont_in,
        ) # (B,)
        return logits, labels


    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_idx = 0
        for batch in self.train_bar:
            logits, labels = self.forward_pass(batch)
            loss = self.loss_fn(logits, labels.float()) # (B,)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        
            total_loss += loss.item()
            self.train_bar.set_postfix({
                    "Loss":  f"{loss.item():.4f}",
                })
            
            if batch_idx % 5 == 0 and self.metrics.wandb_run:
                    self.metrics.wandb_run.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": self.metrics.current_epoch,
                    }, commit=True)
            batch_idx += 1
        return {"loss": total_loss / len(self.train_loader)}
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        batch_idx = 0
        for batch in self.val_bar:
            logits, labels = self.forward_pass(batch)
            loss = self.val_loss_fn(logits, labels.float()) # (B,)
            self.metrics.update_binary_classification(logits, labels)
            total_loss += loss.item()
            self.val_bar.set_postfix({
                    "Loss":  f"{loss.item():.4f}",
                })
            batch_idx += 1
            if batch_idx % 5 == 0 and self.metrics.wandb_run:
                    self.metrics.wandb_run.log({
                        "val_loss": loss.item(),
                        "epoch": self.metrics.current_epoch,
                    }, commit=True)

        return {"loss": total_loss / len(self.val_loader)}
    
