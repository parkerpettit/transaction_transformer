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
import wandb
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

    # ----------------------- W&B artifact per-epoch logging ---------------------- #
    def _build_meta(self, schema: FieldSchema, config: ModelConfig, epoch: int, global_step: int, val_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = {
            "stage": self.stage,
            "schema_hash": self._hash_schema(schema),
            "config_hash": self._hash_model_config(config),
            "git_commit": self._git_commit(),
            "created_at": time.time(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if val_metrics:
            # Keep numeric metrics only for readability
            meta["metrics"] = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
        return meta

    def log_epoch_artifact(
        self,
        wandb_run: Any,
        backbone_path: Path,
        head_path: Path,
        meta: Dict[str, Any],
        epoch: int,
        is_best: bool,
        training_mode: str,
    ) -> None:
        if wandb_run is None:
            return
        # Simple, stable artifact naming for easy downstream consumption
        # Pretrain: always log to a single collection name so finetune can "use_artifact('pretrained-backbone:best')"
        if self.stage == "pretrain":
            base_name = f"pretrained-backbone-{training_mode}"
        else:
            base_name = f"finetuned-model-{training_mode}"
        artifact = wandb.Artifact(base_name, type="model", metadata=meta)
        artifact.add_file(str(backbone_path), name="backbone.pt")
        artifact.add_file(str(head_path), name=("pretrain_head.pt" if self.stage == "pretrain" else "clf_head.pt"))
        # Optionally attach stage logs to the model artifact (pretrain)
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
        # Keep aliases extremely simple: track only latest and best
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
        print(f"[CheckpointManager] Logging W&B artifact {base_name} with aliases {aliases}")
        wandb_run.log_artifact(artifact, aliases=aliases)

    def save_and_log_epoch(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        schema: FieldSchema,
        config: ModelConfig,
        epoch: int,
        global_step: int,
        val_metrics: Optional[Dict[str, Any]],
        wandb_run: Optional[Any],
        is_best: bool,
    ) -> None:
        """Overwrite local exports and log a versioned W&B artifact for this epoch."""
        # Overwrite local files to keep disk usage stable (last)
        backbone_path, head_path = self.save_export_weights(backbone, head, schema, config, best=False)
        # If this is the best so far, also persist a stable 'best' local copy for downstream finetuning fallbacks
        if is_best:
            try:
                self.save_export_weights(backbone, head, schema, config, best=True)
            except Exception as e:
                print(f"[CheckpointManager] Warning: failed to save local 'best' exports: {e}")
        meta = self._build_meta(schema, config, epoch, global_step, val_metrics)
        # Log to W&B with epoch aliases
        self.log_epoch_artifact(
            wandb_run,
            backbone_path,
            head_path,
            meta,
            epoch,
            is_best,
            training_mode=config.training.model_type,
        )

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

    

    # ----------------------- Artifact-based loading helpers ---------------------- #
    @staticmethod
    def load_backbone_from_artifact(
        artifact_ref: str,
        backbone: torch.nn.Module,
        wandb_run: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Download a W&B artifact (e.g., 'entity/project/pretrain-<runid>:best') and load backbone.

        If a W&B run is provided, uses run.use_artifact so the artifact is attached as an input
        dependency to the run, ensuring lineage reflects pretrain -> finetune. Otherwise falls
        back to the public API download which does not create lineage.
        """
        if wandb_run is not None:
            art = wandb_run.use_artifact(artifact_ref)
            artifact_dir = Path(art.download())
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact_ref, type="model")
            artifact_dir = Path(artifact.download())
        payload = torch.load(str(artifact_dir / "backbone.pt"), map_location="cpu", weights_only=False)
        backbone.load_state_dict(payload["state_dict"], strict=True)
        print(f"[CheckpointManager] Loaded backbone from artifact {artifact_ref}")
        return payload.get("meta", {})

    # Discovery helpers removed to simplify the flow. We now rely on a single
    # straightforward rule in finetune: if W&B is enabled and use_local_inputs=false,
    # auto-load the most recent pretrain run's best (or latest) artifact by run id.
