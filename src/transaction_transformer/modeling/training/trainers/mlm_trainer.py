# """
# Masked Language Model trainer for transaction transformer.
# """

# from typing import Dict, Any, Optional
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from transaction_transformer.modeling.training.trainers.base_trainer import BaseTrainer
# import wandb


# class MLMTrainer(BaseTrainer):
#     """Trainer for Masked Language Model training."""
    
#     def __init__(
#         self,
#         model: nn.Module,
#         config: Dict[str, Any],
#         device: torch.device,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         optimizer: torch.optim.Optimizer,
#         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#         wandb_run: Optional[Any] = None
#     ):
#         super().__init__(model, device, train_loader, val_loader, optimizer, scheduler)
#         self.wandb_run = wandb_run
#         self.criterion = self._build_criterion()
#         self.mask_prob = config.get("mask_prob", 0.15)
    
#     def _build_criterion(self) -> nn.Module:
#         """Build loss function for MLM training."""
#         return UnifiedTabularLoss(ignore_index=-100)
    
#     def train_epoch(self) -> Dict[str, float]:
#         """Train for one epoch."""
#         self.model.train()
#         total_loss = 0.0
#         num_batches = 0
        
#         for batch in self.train_loader:
#             # Move batch to device
#             batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
#                     for k, v in batch.items()}
            
#             # Forward pass
#             outputs = self.model(
#                 cat=batch["cat"],
#                 cont=batch["cont"],
#                 causal=False  # MLM uses bidirectional attention
#             )
            
#             # Compute loss
#             loss = self.compute_loss(batch, outputs)
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             if self.scheduler:
#                 self.scheduler.step()
            
#             total_loss += loss.item()
#             num_batches += 1
            
#             # Log to wandb
#             if self.wandb_run:
#                 self.wandb_run.log({
#                     "train_loss": loss.item(),
#                     "learning_rate": self.optimizer.param_groups[0]['lr']
#                 })
        
#         return {"loss": total_loss / num_batches}
    
#     def validate_epoch(self) -> Dict[str, float]:
#         """Validate for one epoch."""
#         self.model.eval()
#         total_loss = 0.0
#         num_batches = 0
        
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 # Move batch to device
#                 batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
#                         for k, v in batch.items()}
                
#                 # Forward pass
#                 outputs = self.model(
#                     cat=batch["cat"],
#                     cont=batch["cont"],
#                     causal=False  # MLM uses bidirectional attention
#                 )
                
#                 # Compute loss
#                 loss = self.compute_loss(batch, outputs)
                
#                 total_loss += loss.item()
#                 num_batches += 1
        
#         return {"val_loss": total_loss / num_batches}
    
#     def compute_loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Compute loss for MLM training using unified loss."""
#         # Prepare labels and probability distributions
#         labels = {}
#         labels_probs = {}
        
#         # For MLM training, outputs should be a dict of logits per field
#         # and batch should contain labels_cat_probs and labels_cont_probs
        
#         # Add categorical probability distributions
#         if "labels_cat_probs" in batch:
#             labels_probs.update({
#                 f"cat_{i}": batch["labels_cat_probs"][..., i, :] 
#                 for i in range(batch["labels_cat_probs"].size(-2))
#             })
        
#         # Add continuous probability distributions  
#         if "labels_cont_probs" in batch:
#             labels_probs.update({
#                 f"cont_{i}": batch["labels_cont_probs"][..., i, :]
#                 for i in range(batch["labels_cont_probs"].size(-2))
#             })
        
#         # For compatibility, also add integer labels
#         if "labels_cat" in batch:
#             labels.update({
#                 f"cat_{i}": batch["labels_cat"][..., i]
#                 for i in range(batch["labels_cat"].size(-1))
#             })
        
#         if "labels_cont" in batch:
#             labels.update({
#                 f"cont_{i}": batch["labels_cont"][..., i]
#                 for i in range(batch["labels_cont"].size(-1))
#             })
        
#         return self.criterion(outputs, labels, labels_probs)
    
#     def _prepare_batch(self, batch: tuple) -> tuple:
#         """Prepare batch for MLM training."""
#         # TODO: Implement batch preparation if needed
#         return batch
    
#     def _apply_masking(self, inputs: torch.Tensor) -> tuple:
#         """Apply masking to inputs for MLM training."""
#         # TODO: Implement masking logic if needed
#         # Note: Masking is handled by the MLM collator
#         pass 