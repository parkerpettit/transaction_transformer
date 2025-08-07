"""
Abstract base trainer class for transaction transformer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schema = schema
        self.config = config
        # Initialize components
        if self.config.model.mode == "pretrain":
            self.checkpoint_manager = CheckpointManager(self.config.model.pretrain_checkpoint_dir) 

        else:
            self.checkpoint_manager = CheckpointManager(self.config.model.finetune_checkpoint_dir) 
        self.metrics = MetricsTracker(ignore_index=self.config.model.data.ignore_idx)
        self.metrics.wandb_run = wandb.init(project=config.metrics.wandb_project, name=config.metrics.run_name, config=config.to_dict(), tags=[config.model.training.model_type])
        self.metrics.class_names = self.schema.cat_features + self.schema.cont_features if config.model.mode == "pretrain" else ["non-fraud", "fraud"]
        self.patience = config.model.training.early_stopping_patience
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.bar_fmt = (
        "{l_bar}{bar:10}| "         # visual bar
        "{n_fmt}/{total_fmt} batches "  # absolute progress
        "({percentage:3.0f}%) | "   # %
        "elapsed: {elapsed} | ETA: {remaining} | "  # timing
        "{rate_fmt} | "             # batches / sec
        "{postfix}"                 # losses go here
    )

    @abstractmethod
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
    def compute_loss(self, logits: Dict[str, torch.Tensor], labels_cat: torch.Tensor, labels_cont: torch.Tensor) -> torch.Tensor:
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
            self.train_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", bar_format=self.bar_fmt, leave=True)
            train_metrics = self.train_epoch()
            self.metrics.end_epoch(self.current_epoch, "train")

            # Validate for one epoch
            self.metrics.start_epoch()
            self.val_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", bar_format=self.bar_fmt, leave=True)
            val_metrics = self.validate_epoch()
            self.metrics.end_epoch(self.current_epoch, "val")
            
            # Save checkpoint if validation loss improves
            if val_metrics.get("loss", float('inf')) < self.best_val_loss:
                print(f"New best validation loss: {val_metrics.get('loss', 0):.4f}")
                self.best_val_loss = val_metrics["loss"]
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.schema, 
                    self.config.model, 
                    wandb_run=self.metrics.wandb_run,
                    name=f"{self.config.model.training.model_type}_{self.config.model.mode}_best_model.pt"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            print(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
            print("-" * 50)
            
            


    def label_smoothing(self, targets: torch.Tensor, num_classes: int, epsilon: float = 0.1) -> torch.Tensor:
        """
        Apply label smoothing to categorical targets.
        Args:
            targets: (N,) int64 tensor of class indices (ground truth, 0 <= v < num_classes)
            num_classes: int, number of classes (q_j)
            epsilon: float, smoothing parameter (epsilon)
        Returns:
            smoothed: (N, num_classes) float tensor, each row is a smoothed probability vector
        Implements:
            [p(v)]_l = 1-epsilon if l==v else epsilon/(num_classes-1)
        """
        N = targets.shape[0]
        smoothed = torch.full((N, num_classes), epsilon / (num_classes - 1), device=targets.device, dtype=torch.float32)
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)
        return smoothed

    def neighbor_label_smoothing(
        self,
        targets: torch.Tensor,
        num_bins: int,
        epsilon: float = 0.1,
        neighborhood: int = 5,
    ) -> torch.Tensor:
        """
        Apply neighborhood label smoothing to quantized (binned) targets.
        Args:
            targets: (N,) int64 tensor of bin indices (ground truth, 0 <= b < num_bins)
            num_bins: int, number of bins (V_j)
            epsilon: float, smoothing parameter (epsilon)
            neighborhood: int, neighborhood size on each side (default 5, so total 10 neighbors)
        Returns:
            smoothed: (N, num_bins) float tensor, each row is a smoothed probability vector
        Implements:
            [p(v)]_l = 1-epsilon if l==b
                       epsilon/10 if l in [b-5, ..., b+5] and l != b
                       0 otherwise
        """
        N = targets.shape[0]
        smoothed = torch.zeros((N, num_bins), device=targets.device, dtype=torch.float32)
        # Main bin
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)
        # Neighborhood bins (excluding main bin)
        for offset in range(-neighborhood, neighborhood + 1):
            if offset == 0:
                continue
            neighbor_idx = targets + offset
            valid = (neighbor_idx >= 0) & (neighbor_idx < num_bins)
            rows = torch.arange(N, device=targets.device)[valid]
            cols = neighbor_idx[valid]
            smoothed[rows, cols] = epsilon / (2 * neighborhood)
        return smoothed
    
