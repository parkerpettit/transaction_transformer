"""
Abstract base trainer class for transaction transformer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .checkpoint_manager import CheckpointManager
from .metrics import MetricsTracker
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import Config
import wandb
from tqdm import tqdm


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

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
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(enabled=config.model.training.use_amp) # type: ignore
        self.autocast = torch.autocast(device_type=self.device.type, enabled=config.model.training.use_amp)
        self.schema = schema
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Initialize components
        if self.config.model.mode == "pretrain":
            self.checkpoint_manager = CheckpointManager(
                self.config.model.pretrain_checkpoint_dir, stage="pretrain"
            )
        else:
            self.checkpoint_manager = CheckpointManager(
                self.config.model.finetune_checkpoint_dir, stage="finetune"
            )
        self.metrics = MetricsTracker(ignore_index=self.config.model.data.ignore_idx)
        self.metrics.wandb_run = wandb.run
        self.metrics.class_names = (
            self.schema.cat_features + self.schema.cont_features
            if self.config.model.mode == "pretrain"
            else ["non-fraud", "fraud"]
        )
        self.patience = config.model.training.early_stopping_patience
        # Training state
        self.current_epoch = 0
        self.current_step = 0   
        self.best_val_loss = float("inf")
        self.bar_fmt = (
            "{l_bar}{bar:10}| "  # visual bar
            "{n_fmt}/{total_fmt} batches "  # absolute progress
            "({percentage:3.0f}%) | "  # %
            "elapsed: {elapsed} | ETA: {remaining} | "  # timing
            "{rate_fmt} | "  # batches / sec
            "{postfix}"  # losses go here
        )
        



    @abstractmethod
    def forward_pass(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a batch."""
        pass

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        labels_cat: torch.Tensor,
        labels_cont: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        pass

    def train(self, num_epochs: int) -> None:
        """Main training loop."""
        patience_counter = 0
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1  
            self.metrics.current_epoch = self.current_epoch
            # Train for one epoch
            self.metrics.start_epoch()
            self.train_bar = tqdm(
                self.train_loader,
                desc=f"Training Epoch {self.current_epoch}",
                bar_format=self.bar_fmt,
                leave=True,
            )
            train_metrics = self.train_epoch()
            self.metrics.end_epoch(self.current_epoch, "train")

            # Validate for one epoch (guard empty loaders)
            self.metrics.start_epoch()
            self.val_bar = tqdm(
                self.val_loader,
                desc=f"Validation Epoch {self.current_epoch}",
                bar_format=self.bar_fmt,
                leave=True,
            )
            val_metrics = self.validate_epoch()
            self.metrics.end_epoch(self.current_epoch, "val")
            if not val_metrics or (
                isinstance(val_metrics.get("loss", None), float)
                and (val_metrics["loss"] != val_metrics["loss"])
            ):
                self.logger.warning(
                    "[Trainer] Warning: validation produced no batches or NaN loss. Check your dataset/splits."
                )

            # Track best and log to W&B every epoch with artifact versioning
            is_best = False
            current_val_loss = val_metrics.get("loss", float("inf"))
            if current_val_loss < self.best_val_loss:
                self.logger.info(f"New best validation loss: {current_val_loss:.4f}")
                self.best_val_loss = current_val_loss
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(
                        f"Early stopping triggered after {self.current_epoch} epochs."
                    )
                    break

            # Save local exports (overwrite) and log a W&B artifact version for this epoch
            backbone = getattr(self.model, "backbone", None)
            head = getattr(self.model, "head", None)
            print(backbone)
            print(head)
            if backbone is not None and head is not None:
                try:
                    self.checkpoint_manager.save_and_log_epoch(
                        backbone=backbone,
                        head=head,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=self.current_epoch,
                        wandb_run=self.metrics.wandb_run,
                        is_best=is_best,
                        model_type=self.config.model.training.model_type,
                    )
                except Exception as e:
                    self.logger.warning(f"[Trainer] Warning: failed to save/log epoch artifact: {e}")

            # Print progress
            self.logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            self.logger.info(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            self.logger.info(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
            self.logger.info("-" * 50)
        wandb.finish()
