"""
chkpt.py
~~~~~~~~
Light-weight checkpoint helpers used by the training loop.  Keeps *one* file
(current run) plus automatically renames incompatible old checkpoints so they
are never overwritten by accident.

Any module may simply::

    from chkpt import save_ckpt, load_ckpt
"""

from __future__ import annotations
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim

from config import ModelConfig   # only for typing / pretty storage

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  save
# ──────────────────────────────────────────────────────────────────────────────
def save_ckpt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val: float,
    path: str | Path,
    cat_features: List[str],
    cont_features: List[str],
    cfg: ModelConfig,
) -> None:
    """
    Persist training state to *path* and sanity-check that it was written.
    """
    path = Path(path)
    log.info("Saving checkpoint → %s (epoch %d)", path, epoch)

    torch.save(
        {
            "epoch":        epoch,
            "best_val":     best_val,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "cat_features": cat_features,
            "cont_features": cont_features,
            "config":       asdict(cfg),
        },
        path,
    )

    # quick verification
    cp = torch.load(path, map_location="cpu", weights_only=False)
    assert cp["epoch"] == epoch, "checkpoint epoch mismatch – write error?"

    sz = path.stat().st_size / 1_048_576
    log.info("Checkpoint written (%.1f MB, %s)", sz, time.ctime(path.stat().st_mtime))


# ──────────────────────────────────────────────────────────────────────────────
#  load / resume
# ──────────────────────────────────────────────────────────────────────────────
def load_ckpt(
    path: str | Path,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    cat_features: List[str],
    cont_features: List[str],
) -> Tuple[float, int]:
    """
    Returns
    -------
    best_val : float
        Best validation loss stored in checkpoint; inf if starting from scratch.
    start_epoch : int
        Epoch index to continue with (0-based). 0 means train from scratch.
    """
    path = Path(path)
    log.info("↺  Looking for checkpoint in %s", path)

    if not path.exists():
        log.info("No checkpoint found – starting fresh.")
        return float("inf"), 0

    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_cat, saved_cont = ckpt.get("cat_features", []), ckpt.get("cont_features", [])

    if saved_cat != cat_features or saved_cont != cont_features:
        # ── feature lists changed – archive old file ───────────────────────
        log.warning("Feature set changed since last run – archiving old checkpoint.")
        if saved_cat != cat_features:
            log.warning("   categorical: %s → %s", saved_cat, cat_features)
        if saved_cont != cont_features:
            log.warning("   continuous  : %s → %s", saved_cont, cont_features)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = path.with_name(f"{path.stem}_old_{ts}{path.suffix}")
        path.rename(archive_name)
        log.info("Old checkpoint moved to %s", archive_name)

        return float("inf"), 0

    # ── safe to resume ─────────────────────────────────────────────────────
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    best_val = ckpt.get("best_val", float("inf"))
    start_ep = ckpt.get("epoch", 0) + 1

    log.info("Resumed from epoch %d  (best_val %.4f)", start_ep, best_val)
    return best_val, start_ep


# utils_cfg.py
import yaml, argparse, pathlib

def load_cfg(path: str | pathlib.Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge(cli_args: argparse.Namespace, file_dict: dict) -> argparse.Namespace:
    """File values are defaults; CLI flags override if given."""
    merged = vars(cli_args).copy()
    merged = {k: (v if v is not None else file_dict.get(k)) for k, v in merged.items()}
    return argparse.Namespace(**merged)
