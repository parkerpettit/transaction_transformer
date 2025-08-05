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
            
            # For continuous features, we still want neighborhood smoothing
            # So we create soft targets and use CrossEntropyLoss with probabilities
            V = self.schema.cont_binners[name].num_bins
            lbl = labels_f.flatten()  # (B*L,)
            logits_flat = logits_f.flatten(0, 1)  # (B*L, V)
            
            # Create soft targets with neighborhood smoothing
            mask = lbl != self.config.model.data.ignore_idx
            if mask.any():
                # Create soft target tensor
                soft_targets = torch.zeros_like(logits_flat)  # (B*L, V)
                soft_targets[~mask] = 0  # Ignored positions stay zero
                
                # Apply neighborhood smoothing for valid positions
                valid_indices = torch.where(mask)[0]
                valid_labels = lbl[mask]
                for i, (idx, label) in enumerate(zip(valid_indices, valid_labels)):
                    soft_targets[idx] = self._create_neighborhood_distribution(int(label.item()), V, epsilon=0.1, neighborhood=5)
                
                # CrossEntropyLoss can handle soft targets!
                total_loss += self.cont_loss_fn(logits_flat, soft_targets)
        
        return total_loss / self.total_features
    
    def _create_neighborhood_distribution(self, target_bin: int, num_bins: int, epsilon: float = 0.1, neighborhood: int = 5) -> torch.Tensor:
        """Create a soft probability distribution with neighborhood smoothing."""
        probs = torch.zeros(num_bins, device=self.device)
        
        # Main bin gets most of the probability mass
        probs[target_bin] = 1.0 - epsilon
        
        # Distribute epsilon among neighbors
        neighbor_count = 0
        for offset in range(1, neighborhood + 1):
            if target_bin - offset >= 0:
                neighbor_count += 1
            if target_bin + offset < num_bins:
                neighbor_count += 1
        
        if neighbor_count > 0:
            neighbor_prob = epsilon / neighbor_count
            for offset in range(1, neighborhood + 1):
                if target_bin - offset >= 0:
                    probs[target_bin - offset] = neighbor_prob
                if target_bin + offset < num_bins:
                    probs[target_bin + offset] = neighbor_prob
        
        return probs




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
                batch_idx += 1

        return {"val_loss": total_loss / num_batches}