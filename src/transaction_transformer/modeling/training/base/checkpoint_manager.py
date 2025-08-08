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
    ) -> None:
        if wandb_run is None:
            return
        # Versioned artifact name per run; aliases distinguish epochs
        base_name = f"{self.stage}-{wandb_run.id}"
        artifact = wandb.Artifact(base_name, type="model", metadata=meta)
        artifact.add_file(str(backbone_path), name="backbone.pt")
        artifact.add_file(str(head_path), name=("pretrain_head.pt" if self.stage == "pretrain" else "clf_head.pt"))
        aliases = [f"epoch-{epoch:04d}", "latest"]
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
        self.log_epoch_artifact(wandb_run, backbone_path, head_path, meta, epoch, is_best)

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

    # ----------------------- Discovery helpers ---------------------- #
    @staticmethod
    def find_latest_stage_artifact_ref(
        entity: str,
        project: str,
        stage: str = "pretrain",
        prefer_alias: str = "best",
    ) -> Optional[str]:
        """Find the most recent W&B model artifact for a given stage in a project.

        Strategy:
          1) Find the latest run with matching jobType ("debug-<stage>") and construct
             the artifact ref "<entity>/<project>/<stage>-<run.id>:<prefer_alias>".
          2) If the preferred alias is missing, fall back to ":latest" for that run.
          3) If that fails, scan all model artifacts named f"{stage}-*" and pick the
             most recently updated one with the preferred alias, else any latest.
        Returns a fully-qualified artifact ref string or None.
        """
        try:
            api = wandb.Api()
            # Step 1: try to locate the most recent run for this stage
            runs = api.runs(f"{entity}/{project}", filters={"jobType": f"debug-{stage}"})
            if len(runs) > 0:
                # Sort by created_at/start_time descending
                def _run_ts(r: Any) -> float:
                    import datetime as _dt
                    val = getattr(r, "created_at", None) or getattr(r, "start_time", None)
                    if isinstance(val, (int, float)):
                        return float(val)
                    if isinstance(val, _dt.datetime):
                        return val.timestamp()
                    if isinstance(val, str):
                        try:
                            # W&B often returns ISO strings
                            return _dt.datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                        except Exception:
                            return 0.0
                    return 0.0
                runs_sorted = sorted(runs, key=_run_ts, reverse=True)
                latest_run = runs_sorted[0]
                base = f"{entity}/{project}/{stage}-{latest_run.id}"
                # Prefer alias first
                candidate = f"{base}:{prefer_alias}"
                try:
                    _ = api.artifact(candidate, type="model")
                    return candidate
                except Exception:
                    # Fall back to latest for that run
                    candidate_latest = f"{base}:latest"
                    try:
                        _ = api.artifact(candidate_latest, type="model")
                        return candidate_latest
                    except Exception:
                        pass

            # Step 2: global scan across collections named stage-*
            atype = api.artifact_type(type_name="model", project=f"{entity}/{project}")
            collections = atype.collections()
            best_artifacts = []
            latest_artifacts = []
            for coll in collections:
                if not coll.name.startswith(f"{stage}-"):
                    continue
                versions_fn = getattr(coll, "versions", None)
                version_list_iter = []
                if callable(versions_fn):
                    try:
                        for _art in versions_fn():  # type: ignore[misc, call-arg]
                            version_list_iter.append(_art)
                    except Exception:
                        version_list_iter = []
                for art in version_list_iter:
                    # aliases may be strings or objects with .name
                    alias_names = []
                    for a in getattr(art, "aliases", []):
                        name = getattr(a, "name", None)
                        if isinstance(a, str):
                            alias_names.append(a)
                        elif isinstance(name, str):
                            alias_names.append(name)
                    if prefer_alias in alias_names:
                        best_artifacts.append(art)
                    if "latest" in alias_names:
                        latest_artifacts.append(art)
            # Prefer artifacts tagged with the preferred alias
            def _art_time(a: Any) -> float:
                import datetime as _dt
                val = getattr(a, "updated_at", None) or getattr(a, "created_at", None)
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, _dt.datetime):
                    return val.timestamp()
                if isinstance(val, str):
                    try:
                        return _dt.datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        return 0.0
                return 0.0

            if best_artifacts:
                best_artifacts.sort(key=_art_time, reverse=True)
                return getattr(best_artifacts[0], "qualname", None)
            if latest_artifacts:
                latest_artifacts.sort(key=_art_time, reverse=True)
                return getattr(latest_artifacts[0], "qualname", None)
        except Exception as e:
            print(f"[CheckpointManager] Warning: failed to find latest artifact: {e}")
        return None
