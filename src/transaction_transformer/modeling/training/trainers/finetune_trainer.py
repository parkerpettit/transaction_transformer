"""
Finetuning trainer for transaction transformer.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..base.base_trainer import BaseTrainer
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.data.preprocessing import FieldSchema

class FinetuneTrainer(BaseTrainer):
    """Trainer for finetuning models on specific tasks."""
    
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
        super().__init__(model, schema, config, device, train_loader, val_loader, optimizer, scheduler)
        # self.criterion = self._build_criterion()
    
    # def _build_criterion(self) -> nn.Module:
    #     """Build loss function for finetuning."""
    #     # TODO: Implement criterion building
    #     pass
    
    # def train_epoch(self) -> Dict[str, float]:
    #     """Train for one epoch."""
    #     # TODO: Implement finetuning epoch
    #     pass
    
    # def validate_epoch(self) -> Dict[str, float]:
    #     """Validate for one epoch."""
    #     # TODO: Implement validation epoch
    #     pass
    
    # def compute_loss(self, batch: tuple, outputs: Any) -> torch.Tensor:
    #     """Compute loss for finetuning."""
    #     # TODO: Implement loss computation
    #     pass
    
    # def _prepare_batch(self, batch: tuple) -> tuple:
    #     """Prepare batch for training."""
    #     # TODO: Implement batch preparation
    #     pass
    
    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load pretrained weights for finetuning."""
    #     # TODO: Implement pretrained weight loading
    #     pass 