"""
Checkpoint management for transaction transformer. Checkpoints are overwritten each time they get saved. They will be uploaded
to wandb, so we don't need to save all checkpoints to disk.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig


class CheckpointManager:
    """Manages model checkpoint saving and loading using state_dicts."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        # Ensure directory exists to avoid save errors
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        schema: FieldSchema,
        config: ModelConfig,
        wandb_run: Optional[Any] = None,
        name: str = "checkpoint.pt",
    ) -> None:
        """Save a training checkpoint using state_dicts."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "schema": schema,
            "config": config,
        }

        print(f"Saving checkpoint to {self.checkpoint_dir / name}")
        torch.save(checkpoint, self.checkpoint_dir / name)
        print(f"Checkpoint saved to {self.checkpoint_dir / name}")
        if wandb_run:
            print(f"Uploading checkpoint to wandb")
            wandb_run.log_artifact(self.checkpoint_dir / name, type="model")
            print(f"Checkpoint uploaded to wandb")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: Optional[str] = None,
    ) -> Tuple[
        torch.nn.Module,
        Optional[torch.optim.Optimizer],
        Optional[torch.optim.lr_scheduler._LRScheduler],
        int,
    ]:
        """
        Load a training checkpoint. Loads state_dicts into the provided model, optimizer, and scheduler.
        Returns model, optimizer, scheduler, and current epoch.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        return model, optimizer, scheduler, epoch
