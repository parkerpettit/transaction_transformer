"""
Checkpoint management for transaction transformer.

"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import wandb
import logging


class CheckpointManager:
    """Manages exportable weights and atomic resume checkpoints."""

    def __init__(self, checkpoint_dir: str, stage: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage  # "pretrain" or "finetune"
        self.logger = logging.getLogger(__name__)
    # -------------------------- export weights ---------------------------- #
    def save_export_weights(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> Tuple[Path, Path, Path]:
        backbone_path = self.checkpoint_dir / f"backbone.pt"
        head_path = self.checkpoint_dir / f"head.pt"
        torch.save({"state_dict": backbone.state_dict()}, backbone_path)
        torch.save({"state_dict": head.state_dict()}, head_path)
        optimizer_scheduler_path = self.checkpoint_dir / "optimizer_scheduler.pt"
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, optimizer_scheduler_path)
        print(
            f"[CheckpointManager] Saved exportable weights -> {backbone_path.name}, {head_path.name}, {optimizer_scheduler_path.name}"
        )
        return backbone_path, head_path, optimizer_scheduler_path

    def log_epoch_artifact(
        self,
        wandb_run: Any,
        backbone_path: Path,
        head_path: Path,
        optimizer_scheduler_path: Path,
        epoch: int,
        is_best: bool,
        model_type: str,
        head_type: str,
        is_lora: bool,
    ) -> None:
        if wandb_run is None:
            return
        base_name = f"{self.stage}-{model_type}"
        description = f"Epoch {epoch} of {self.stage} training for {model_type} model with {head_type} head. Is LoRA: {is_lora}."
        artifact = wandb.Artifact(base_name, type="model", metadata={"model_type": model_type, "epoch": epoch, "head_type": head_type, "is_lora": is_lora}, description=description)
        artifact.add_file(str(backbone_path), name="backbone.pt")
        artifact.add_file(
            str(head_path),
            name="head.pt",
        )
        artifact.add_file(str(optimizer_scheduler_path), name="optimizer_scheduler.pt")
        # Attach stage logs to the model artifact 
        if self.stage == "pretrain":
            log_file = os.environ.get("TT_PRETRAIN_LOG_FILE")
            if log_file and Path(log_file).exists():
                try:
                    artifact.add_file(str(log_file), name="pretrain.log")
                except Exception:
                    pass
        elif self.stage == "finetune":
            log_file = os.environ.get("TT_FINETUNE_LOG_FILE")
            if log_file and Path(log_file).exists():
                try:
                    artifact.add_file(str(log_file), name="finetune.log")
                except Exception:
                    pass
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
        
        
        wandb_run.log_artifact(artifact, aliases=aliases)

    def save_and_log_epoch(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        wandb_run: Optional[Any],
        is_best: bool,
        model_type: str,
        head_type: str,
        is_lora: bool,
    ) -> None:
        """Overwrite local exports and log a versioned W&B artifact for this epoch."""
        # Overwrite local files to keep disk usage stable (last)
        backbone_path, head_path, optimizer_scheduler_path = self.save_export_weights(
            backbone, head, optimizer, scheduler
        )

        # Log to W&B with epoch aliases
        self.log_epoch_artifact(
            wandb_run,
            backbone_path,
            head_path,
            optimizer_scheduler_path,
            epoch,
            is_best,
            model_type=model_type,
            head_type=head_type,
            is_lora=is_lora
        )

    # -------------------------- loading helpers --------------------------- #
    def load_export_backbone(self, path: str, backbone: torch.nn.Module) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        backbone.load_state_dict(state, strict=True)
        self.logger.info(f"[CheckpointManager] Loaded backbone export from {path}")
        return payload.get("meta", {})

    def load_export_head(self, path: str, head: torch.nn.Module) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(payload["state_dict"], strict=True)
        self.logger.info(f"[CheckpointManager] Loaded head export from {path}")
        return payload.get("meta", {})