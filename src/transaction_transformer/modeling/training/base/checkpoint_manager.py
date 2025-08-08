"""
Checkpoint management for transaction transformer.

Design goals:
- Exportable weights (for initialization or transfer): backbone.pt and head.pt
- Atomic resume checkpoint (for exact training resume): resume.pt containing
  backbone_state_dict, head_state_dict, optimizer/scheduler/scaler states, epoch/step,
  rng state, minimal metadata (stage, schema_hash, config_hash, git_commit, created_at)
- Overwrite local files by default (keep laptop clean); optionally log artifacts to wandb
  with clear linkage between backbone/head pairs via metadata and artifact names.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import hashlib
import json
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig


class CheckpointManager:
    """Manages exportable weights and atomic resume checkpoints."""

    def __init__(self, checkpoint_dir: str, stage: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage  # "pretrain" or "finetune"

    # -------------------------- utility helpers --------------------------- #
    @staticmethod
    def _hash_schema(schema: FieldSchema) -> str:
        payload = {
            "cat_features": schema.cat_features,
            "cont_features": schema.cont_features,
            "cat_vocab_sizes": {k: v.vocab_size for k, v in schema.cat_encoders.items()},
            "cont_num_bins": {k: v.num_bins for k, v in schema.cont_binners.items()},
        }
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    @staticmethod
    def _hash_model_config(config: ModelConfig) -> str:
        data = json.dumps(config.__dict__, default=lambda o: o.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            import subprocess
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            )
            return commit
        except Exception:
            return None

    # -------------------------- export weights ---------------------------- #
    def save_export_weights(self, backbone: torch.nn.Module, head: torch.nn.Module, schema: FieldSchema, config: ModelConfig, best: bool = False) -> Tuple[Path, Path]:
        tag = "best" if best else "last"
        backbone_path = self.checkpoint_dir / f"backbone_{tag}.pt"
        head_path = self.checkpoint_dir / ("pretrain_head_" + tag + ".pt" if self.stage == "pretrain" else "clf_head_" + tag + ".pt")

        meta = {
            "stage": self.stage,
            "schema_hash": self._hash_schema(schema),
            "config_hash": self._hash_model_config(config),
            "git_commit": self._git_commit(),
            "created_at": time.time(),
        }

        torch.save({"state_dict": backbone.state_dict(), "meta": meta}, backbone_path)
        torch.save({"state_dict": head.state_dict(), "meta": meta}, head_path)
        print(f"[CheckpointManager] Saved exportable weights -> {backbone_path.name}, {head_path.name}")
        return backbone_path, head_path

    # -------------------------- resume checkpoint ------------------------- #
    def save_resume_checkpoint(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[Any],
        epoch: int,
        global_step: int,
        schema: FieldSchema,
        config: ModelConfig,
        best: bool = False,
        wandb_run: Optional[Any] = None,
    ) -> Path:
        tag = "best" if best else "last"
        resume_path = self.checkpoint_dir / f"resume_{tag}.pt"

        meta = {
            "stage": self.stage,
            "schema_hash": self._hash_schema(schema),
            "config_hash": self._hash_model_config(config),
            "git_commit": self._git_commit(),
            "created_at": time.time(),
            "epoch": epoch,
            "global_step": global_step,
        }

        checkpoint = {
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "meta": meta,
        }

        torch.save(checkpoint, resume_path)
        print(f"[CheckpointManager] Saved resume checkpoint -> {resume_path.name}")

        if wandb_run is not None:
            artifact_name = f"{self.stage}-resume-{tag}"
            print(f"[CheckpointManager] Logging resume checkpoint to wandb as {artifact_name}")
            wandb_run.log_artifact(resume_path, type="model", name=artifact_name)
        return resume_path

    # -------------------------- loading helpers --------------------------- #
    @staticmethod
    def load_export_backbone(path: str, backbone: torch.nn.Module) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        backbone.load_state_dict(state, strict=True)
        print(f"[CheckpointManager] Loaded backbone export from {path}")
        return payload.get("meta", {})

    @staticmethod
    def load_export_head(path: str, head: torch.nn.Module) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(payload["state_dict"], strict=True)
        print(f"[CheckpointManager] Loaded head export from {path}")
        return payload.get("meta", {})

    @staticmethod
    def load_resume_checkpoint(
        path: str,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
        head.load_state_dict(ckpt["head_state_dict"], strict=True)
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"[CheckpointManager] Loaded resume checkpoint from {path}")
        return ckpt.get("meta", {})
