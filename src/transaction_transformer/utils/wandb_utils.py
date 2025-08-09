"""
Centralized Weights & Biases utilities.

This module provides a single entrypoint to initialize W&B and helpers to
attach and download artifacts. All scripts should import from here and pass
the returned `wandb_run` around rather than calling wandb.init() in multiple
places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import wandb


def _project_root() -> Path:
    # utils/ -> transaction_transformer/ -> src/ -> repo_root
    return Path(__file__).resolve().parents[3]


def init_wandb(config: Any, job_type: str, tags: Optional[list[str]] = None) -> Optional[Any]:
    """
    Initialize a single W&B run and return it. Returns None if config disables W&B.

    The function is intentionally minimal. We do NOT change the global step or
    define custom metric steps globally. We simply include "epoch" in every log
    call elsewhere so that charts can use epoch as an x-axis when desired.
    """
    if not getattr(config.metrics, "wandb", False):
        return None

    entity = getattr(config.metrics, "wandb_entity", None)
    run = wandb.init(
        entity=entity,
        project=config.metrics.wandb_project,
        name=config.metrics.run_name,
        job_type=job_type,
        config=config.to_dict() if hasattr(config, "to_dict") else None,
        tags=tags or [],
    )
    return run




def download_artifact(
    wandb_run: Optional[Any],
    artifact_ref: str,
    *,
    type: Optional[str] = None,
    root: Optional[str | Path] = None,
) -> Path:
    """
    Download an artifact by reference and return the local directory path.

    If a run is provided, the artifact is attached as an input to that run.
    Otherwise, uses the public API (no lineage).
    """
    if wandb_run is not None:
        art = wandb_run.use_artifact(artifact_ref)
        adir = Path(art.download(root=str(root) if root is not None else None))
        return adir
    api = wandb.Api()
    art = api.artifact(artifact_ref, type=type) if type is not None else api.artifact(artifact_ref)
    adir = Path(art.download(root=str(root) if root is not None else None))
    return adir


