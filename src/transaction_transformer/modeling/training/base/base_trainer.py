"""
Abstract base trainer class for transaction transformer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .checkpoint_manager import CheckpointManager
from .metrics import MetricsTracker
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig
import wandb
from tqdm import tqdm

class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        schema: FieldSchema,
        config: ModelConfig,
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
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir) 
        self.metrics = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        self.train_bar = tqdm(train_loader, desc="Training", bar_format=self.bar_fmt, leave=True)
        self.val_bar = tqdm(val_loader, desc="Validation", bar_format=self.bar_fmt, leave=True)


    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        pass
    
    @abstractmethod
    def compute_loss(self, batch: tuple, outputs: Any) -> torch.Tensor:
        """Compute loss for a batch."""
        pass
    
    def train(self, num_epochs: int) -> None:
        """Main training loop."""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.train_bar.reset()
            self.val_bar.reset()
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate for one epoch
            val_metrics = self.validate_epoch()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.metrics.log_metrics(metrics)
            
            # Save checkpoint if validation loss improves
            if val_metrics.get("val_loss", float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, "best_model.pt")
            
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            print(f"Val Loss: {val_metrics.get('val_loss', 0):.4f}")
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
    

    def save_checkpoint(self, epoch: int, name: str = "checkpoint.pt") -> None:
        """Save training checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            self.schema, 
            self.config, 
            name=name
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        self.model, self.optimizer, self.scheduler, self.current_epoch = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler
        )
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        self.metrics.log_metrics(metrics)
