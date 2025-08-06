"""
MLM training trainer for transaction transformer.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..base.base_trainer import BaseTrainer
import torch.nn.functional as F
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import Config


class Pretrainer(BaseTrainer):
    """Trainer for pretraining the model."""
    
    def __init__(
        self,
        model: nn.Module,
        schema: FieldSchema,
        config: Config,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__(model, schema, config, device, train_loader, val_loader, optimizer, scheduler)
        # Use CrossEntropyLoss for both AR and MLM - it handles everything!
        # For categorical features: use built-in label smoothing
        self.cat_loss_fn = nn.CrossEntropyLoss(ignore_index=config.model.data.ignore_idx, label_smoothing=0.1)
        # For continuous features: we'll still use custom neighborhood smoothing
        self.cont_loss_fn = nn.CrossEntropyLoss(ignore_index=config.model.data.ignore_idx)
        self.total_features = len(self.schema.cat_features) + len(self.schema.cont_features)
        print(f"Total features: {self.total_features}")
        print(f"Model type: {config.model.training.model_type}")
    
    def compute_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
        # Unified loss computation for both AR and MLM!
        return self.compute_unified_loss(logits, labels_cat, labels_cont)

    def compute_unified_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
        """Unified loss computation for both AR and MLM.
        
        Args:
            logits: dict[name] : (B, L, V_f) - predictions at each position
            labels_cat: (B, L, C) - categorical targets (original for MLM, shifted for AR)
            labels_cont: (B, L, F) - continuous targets (binned)
            
        Returns:
            Average loss across all features
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Handle categorical features with built-in label smoothing
        for f_idx, name in enumerate(self.schema.cat_features):
            logits_f = logits[name]  # (B, L, V)
            labels_f = labels_cat[:, :, f_idx]  # (B, L)
            # CrossEntropyLoss handles ignore_index and label_smoothing automatically!
            total_loss += self.cat_loss_fn(logits_f.flatten(0, 1), labels_f.flatten())
        
        # Handle continuous features with custom neighborhood smoothing
        for f_idx, name in enumerate(self.schema.cont_features):
            logits_f = logits[name]  # (B, L, V)
            labels_f = labels_cont[:, :, f_idx]  # (B, L)
            
            V = self.schema.cont_binners[name].num_bins
            lbl = labels_f.flatten()  # (B*L,)
            logits_flat = logits_f.flatten(0, 1)  # (B*L, V)
            
            mask = lbl != self.config.model.data.ignore_idx
            if mask.any():
                soft_targets = self._vectorized_neighborhood_smoothing(
                    labels=lbl[mask],
                    num_bins=V,
                    epsilon=0.1,
                    neighborhood=5
                )
                
                # CrossEntropyLoss with soft targets needs the full target tensor
                full_soft_targets = torch.zeros_like(logits_flat)
                full_soft_targets[mask] = soft_targets

                total_loss += self.cont_loss_fn(logits_flat, full_soft_targets)
        
        return total_loss / self.total_features
    
    def _vectorized_neighborhood_smoothing(self, labels: torch.Tensor, num_bins: int, epsilon: float, neighborhood: int) -> torch.Tensor:
        """
        Creates a soft-probability distribution with neighborhood smoothing in a vectorized manner.

        Args:
            labels: (N,) tensor of ground-truth bin indices. N is the number of valid (non-ignored) labels.
            num_bins: The total number of bins (V).
            epsilon: The total probability mass to be distributed among neighbors.
            neighborhood: The number of neighbors on each side to receive probability mass.

        Returns:
            (N, V) tensor of soft-probability distributions.
        """
        N = labels.shape[0]
        soft_targets = torch.zeros((N, num_bins), device=self.device)

        # Set the probability for the target bin
        soft_targets.scatter_(1, labels.unsqueeze(1), 1.0 - epsilon)

        # Calculate number of valid neighbors for each label to correctly normalize the probability
        left_neighbors = torch.clamp(labels, max=neighborhood)
        right_neighbors = torch.clamp((num_bins - 1) - labels, max=neighborhood)
        total_neighbors = left_neighbors + right_neighbors
        
        # Avoid division by zero for labels that have no neighbors (shouldn't happen with k>0)
        total_neighbors = torch.max(total_neighbors, torch.tensor(1, device=self.device, dtype=total_neighbors.dtype))
        
        neighbor_prob = epsilon / total_neighbors.float() # (N,)

        row_indices = torch.arange(N, device=self.device)
        # Distribute probability mass to neighbors using a small loop (over k, not N)
        for offset in range(1, neighborhood + 1):
            # Left neighbors
            left_indices = labels - offset
            left_mask = left_indices >= 0
            if left_mask.any():
                 soft_targets[row_indices[left_mask], left_indices[left_mask]] += neighbor_prob[left_mask]

            # Right neighbors
            right_indices = labels + offset
            right_mask = right_indices < num_bins
            if right_mask.any():
                soft_targets[row_indices[right_mask], right_indices[right_mask]] += neighbor_prob[right_mask]

        return soft_targets




    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a batch."""
        cat_in = batch["cat"].to(self.device) # (B, L-1, C) or (B, L, C)
        cont_in = batch["cont"].to(self.device) # (B, L-1, F) or (B, L, F)
        labels_cat = batch["labels_cat"].to(self.device) # (B, C) or (B, L, C)
        labels_cont = batch["labels_cont"].to(self.device) # (B, F) or (B, L, F)
        logits = self.model(
            cat=cat_in,
            cont=cont_in,
        ) # dict of length K = C + F, where each key is a feature name and each value is a (B, V_field) tensor.
        return logits, labels_cat, labels_cont

    
    def train_epoch(self) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0  # accumulate as python float to avoid autograd graph growth
            num_batches = 0
            batch_idx = 0
            for batch in self.train_bar:
                # if batch_idx == 50:
                #     break
                logits, labels_cat, labels_cont = self.forward_pass(batch)
                loss = self.compute_loss(logits, labels_cat, labels_cont)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                if batch_idx % 10 == 0 and self.metrics.wandb_run:
                    self.metrics.wandb_run.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                    }, commit=True)
                
                batch_idx += 1
                if batch_idx % 10 == 0:
                    self.train_bar.set_postfix({
                            "Loss":  f"{loss.item():.4f}",
                        })
            
            return {"loss": total_loss / num_batches,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        batch_idx = 0
        with torch.no_grad():
            for batch in self.val_bar:
                # if batch_idx == 50:
                #     break
                logits, labels_cat, labels_cont = self.forward_pass(batch)
                loss = self.compute_loss(logits, labels_cat, labels_cont)

                targets = {}
                for i, feature_name in enumerate(self.schema.cat_features):
                    targets[feature_name] = labels_cat[:, :, i]  # (B, L) for both AR and MLM
                for i, feature_name in enumerate(self.schema.cont_features):
                    targets[feature_name] = labels_cont[:, :, i]  # (B, L) for both AR and MLM
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
                self.metrics.update_transaction_prediction(logits, targets)
                total_loss += loss.item()
                num_batches += 1

                self.val_bar.set_postfix({
                    "Loss":  f"{loss.item():.4f}",
                })

                if batch_idx == 0:
                    self.metrics.print_sample_predictions(logits, targets, self.schema)
                if batch_idx % 5 == 0 and self.metrics.wandb_run:
                    self.metrics.wandb_run.log({
                        "val_loss": loss.item(),
                    }, commit=True)
                batch_idx += 1

        return {"loss": total_loss / num_batches}