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
        if config.model.training.model_type == "ar":
            self.loss_fn = nn.CrossEntropyLoss()
        elif config.model.training.model_type == "mlm":
            self.loss_fn = nn.KLDivLoss(reduction="batchmean")   # expects log-probs + soft targets
        else:
            raise ValueError(f"Invalid training model_type: {config.model.training.model_type}")
        self.total_features = len(self.schema.cat_features) + len(self.schema.cont_features)
        print(f"Total features: {self.total_features}")
        print(f"Model type: {config.model.training.model_type}")
    
    def compute_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
        if self.config.model.training.model_type == "mlm":
            return self.compute_mlm_loss(logits, labels_cat, labels_cont)
        elif self.config.model.training.model_type == "ar":
            return self.compute_ar_loss(logits, labels_cat, labels_cont)
        else:
            raise ValueError(f"Invalid training model_type: {self.config.model.training.model_type}")

    # ------------------------------------------------------------------ #
    def compute_ar_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
        """Compute loss for autoregressive training using unified loss. 
        labels_cat and labels_cont are the ground truth labels for the categorical and continuous features, respectively, of shape (B, C) and (B, F).
        logits is a dictionary of length K = C + F, where each key is a feature name and each value is a (B, V_feature) tensor of logits. 
        For AR training, we want the model to predict the next transaction, so we use the last position of the output.
        Returns the total loss.

        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # For AR training, we want to predict the next transaction given the context
        # The model outputs logits for the next transaction at the last position
        
        # Handle categorical features
        for i, feature_name in enumerate(self.schema.cat_features):
            feature_logits = logits[feature_name]  # (B, vocab_size) - predict next transaction
            feature_targets = labels_cat[:, i]  # (B,) - target id for this feature
            feature_targets = self.label_smoothing(feature_targets, self.schema.cat_encoders[feature_name].vocab_size)
            total_loss = total_loss + self.loss_fn(feature_logits, feature_targets)
            
        # Handle continuous features  
        for i, feature_name in enumerate(self.schema.cont_features):
            feature_logits = logits[feature_name]  # (B, num_bins) - predict next transaction
            feature_targets = labels_cont[:, i]  # (B,) - binned target id for this feature
            feature_targets = self.neighbor_label_smoothing(feature_targets, self.schema.cont_binners[feature_name].num_bins)
            total_loss = total_loss + self.loss_fn(feature_logits, feature_targets)
        
        return total_loss / self.total_features # average loss per feature

    def compute_mlm_loss(
        self,
        logits: Dict[str, torch.Tensor],       # dict[name] : (B, L, V_f)
        labels_cat: torch.Tensor,              # (B, L, C)  long
        labels_cont: torch.Tensor,             # (B, L, F)  long
        ignore_idx: int = -100,
        eps_cat: float = 0.1,
        eps_cont: float = 0.1,
        neigh: int = 5,
    ) -> torch.Tensor:
        """
        Unified masked-token loss with per-field label smoothing.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # ----------------------------- #
        # categorical fields
        # ----------------------------- #
        total_masked = 0
        for f_idx, name in enumerate(self.schema.cat_features):
            V = self.schema.cat_encoders[name].vocab_size
            lbl     = labels_cat[:, :, f_idx].flatten()                        # (B*L,)
            mask    = lbl != ignore_idx                                        # bool
            if mask.any():
                total_masked += mask.sum().item()
                tgt_ids = lbl[mask]                                            # (N,)
                tgt_prob = self.label_smoothing(tgt_ids, V, epsilon=eps_cat)   # (N,V)
                logp     = F.log_softmax(logits[name].flatten(0, 1)[mask], -1) # (N,V)
                total_loss += self.loss_fn(logp, tgt_prob)

        # ----------------------------- #
        # continuous / binned fields
        # ----------------------------- #
        for f_idx, name in enumerate(self.schema.cont_features):
            V = self.schema.cont_binners[name].num_bins
            lbl     = labels_cont[:, :, f_idx].flatten()
            mask    = lbl != ignore_idx
            if mask.any():
                total_masked += mask.sum().item()
                tgt_ids = lbl[mask]
                tgt_prob = self.neighbor_label_smoothing(
                    tgt_ids, V, epsilon=eps_cont, neighborhood=neigh
                )                                                               # (N,V)
                logp = F.log_softmax(logits[name].flatten(0, 1)[mask], -1)      # (N,V)
                total_loss += self.loss_fn(logp, tgt_prob)

        # average over number of **features** that contributed
        return total_loss / total_masked



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
                    if self.config.model.training.model_type == "ar":
                        targets[feature_name] = labels_cat[:, i]  # (B,)
                    else:
                        targets[feature_name] = labels_cat[:, :, i]  # (B, L)
                for i, feature_name in enumerate(self.schema.cont_features):
                    if self.config.model.training.model_type == "ar":
                        targets[feature_name] = labels_cont[:, i]  # (B,)
                    else:
                        targets[feature_name] = labels_cont[:, :, i]  # (B, L)
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