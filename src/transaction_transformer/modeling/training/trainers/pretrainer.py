"""
MLM training trainer for transaction transformer.
"""

from typing import Dict, Any, Optional, Tuple
import logging
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
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__(
            model,
            schema,
            config,
            device,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
        )
        self.total_features = len(self.schema.cat_features) + len(
            self.schema.cont_features
        )

        print(f"Total features: {self.total_features}")
        print(f"Model type: {config.model.training.model_type}")


    def compute_loss(self, logits, labels_cat, labels_cont) -> torch.Tensor:
        total_sum = torch.tensor(0.0, device=self.device)
        total_valid = torch.tensor(0.0, device=self.device)
        ignore_idx = self.config.model.data.ignore_idx

        # Categorical (hard targets)
        for f_idx, name in enumerate(self.schema.cat_features):
            logits_f = logits[name].flatten(0, 1)         # (B*L, V)
            labels_f = labels_cat[:, :, f_idx].flatten()  # (B*L,)
            valid = labels_f != ignore_idx

            # per-position CE; zeros for ignored when using ignore_index with 'none'
            per_loss = F.cross_entropy(
                logits_f, labels_f,
                reduction="none",
                ignore_index=ignore_idx,
                label_smoothing=0.1,
            )                                             # (B*L,)
            total_sum += per_loss[valid].sum()
            total_valid += valid.sum()

        # Continuous (soft targets)
        for f_idx, name in enumerate(self.schema.cont_features):
            logits_f = logits[name].flatten(0, 1)         # (B*L, V)
            labels_f = labels_cont[:, :, f_idx].flatten() # (B*L,)
            valid = labels_f != ignore_idx
            if valid.any():
                V = self.schema.cont_binners[name].num_bins
                soft_targets = self._vectorized_neighborhood_smoothing(
                    labels=labels_f[valid], num_bins=V, epsilon=0.1, neighborhood=5
                ).to(dtype=logits_f.dtype)                # (N_valid, V)

                # per-position CE with soft/probability targets
                per_loss = F.cross_entropy(
                    logits_f[valid], soft_targets, reduction="none"
                )                                         # (N_valid,)
                total_sum += per_loss.sum()
                total_valid += valid.sum()

        return total_sum / total_valid.clamp_min(1)

    def _vectorized_neighborhood_smoothing(
        self, labels: torch.Tensor, num_bins: int, epsilon: float, neighborhood: int
    ) -> torch.Tensor:
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
        total_neighbors = torch.max(
            total_neighbors,
            torch.tensor(1, device=self.device, dtype=total_neighbors.dtype),
        )

        neighbor_prob = epsilon / total_neighbors.float()  # (N,)

        row_indices = torch.arange(N, device=self.device)
        # Distribute probability mass to neighbors using a small loop (over k, not N)
        for offset in range(1, neighborhood + 1):
            # Left neighbors
            left_indices = labels - offset
            left_mask = left_indices >= 0
            if left_mask.any():
                soft_targets[
                    row_indices[left_mask], left_indices[left_mask]
                ] += neighbor_prob[left_mask]

            # Right neighbors
            right_indices = labels + offset
            right_mask = right_indices < num_bins
            if right_mask.any():
                soft_targets[
                    row_indices[right_mask], right_indices[right_mask]
                ] += neighbor_prob[right_mask]

        return soft_targets

    def forward_pass(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a batch."""
        cat_in = batch["cat"].to(self.device)  # (B, L-1, C) or (B, L, C)
        cont_in = batch["cont"].to(self.device)  # (B, L-1, F) or (B, L, F)
        labels_cat = batch["labels_cat"].to(self.device)  # (B, C) or (B, L, C)
        labels_cont = batch["labels_cont"].to(self.device)  # (B, F) or (B, L, F)
        logits = self.model(
            cat=cat_in,
            cont=cont_in,
        )  # dict of length K = C + F, where each key is a feature name and each value is a (B, V_field) tensor.
        return logits, labels_cat, labels_cont

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0  # accumulate as python float to avoid autograd graph growth
        num_batches = 0
        batch_idx = 0
        max_batches = getattr(self.config.model.training, "max_batches_per_epoch", None)
        for batch in self.train_bar:
            if (
                isinstance(max_batches, int)
                and max_batches > 0
                and batch_idx >= max_batches
            ):
                break

            with self.autocast:
                if batch_idx == 0:
                    print(f"Using {self.config.model.training.use_amp} for batch {batch_idx}")
                logits, labels_cat, labels_cont = self.forward_pass(batch)
                loss = self.compute_loss(logits, labels_cat, labels_cont)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            

            total_loss += loss.item()
            num_batches += 1
            if batch_idx % 10 == 0 and self.metrics.wandb_run:
                self.metrics.wandb_run.log(
                    {
                        "epoch": self.metrics.current_epoch,
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    commit=True,
                )

            batch_idx += 1
            if batch_idx % 10 == 0:
                self.train_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                    }
                )
        self.scheduler.step()
        return {
            "loss": total_loss / num_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        batch_idx = 0
        with torch.no_grad():
            max_batches = getattr(
                self.config.model.training, "max_batches_per_epoch", None
            )
            for batch in self.val_bar:
                if (
                    isinstance(max_batches, int)
                    and max_batches > 0
                    and batch_idx >= max_batches
                ):
                    break
                with self.autocast:
                    logits, labels_cat, labels_cont = self.forward_pass(batch)
                    loss = self.compute_loss(logits, labels_cat, labels_cont)

                targets = {}
                for i, feature_name in enumerate(self.schema.cat_features):
                    targets[feature_name] = labels_cat[
                        :, :, i
                    ]  # (B, L) for both AR and MLM
                for i, feature_name in enumerate(self.schema.cont_features):
                    targets[feature_name] = labels_cont[
                        :, :, i
                    ]  # (B, L) for both AR and MLM
                targets = {k: v.to(self.device) for k, v in targets.items()}

                self.metrics.update_transaction_prediction(logits, targets)
                total_loss += loss.item()
                num_batches += 1

                self.val_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                    }
                )
                if batch_idx % 5 == 0 and self.metrics.wandb_run:
                    self.metrics.wandb_run.log(
                        {
                            "epoch": self.metrics.current_epoch,
                            "val_loss": loss.item(),
                        },
                        commit=True,
                    )
                if batch_idx == 0:
                    self.metrics.print_sample_predictions(logits, targets, self.schema)
                batch_idx += 1

        if num_batches == 0:
            return {"loss": float("nan")}
        return {"loss": total_loss / num_batches}
