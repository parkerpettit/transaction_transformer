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
from model import TransactionModel
import torch
from torch import nn, optim

from config import ModelConfig   # only for typing / pretty storage

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  save
# ──────────────────────────────────────────────────────────────────────────────
def save_ckpt(
    model: nn.Module,
    optim: optim.Optimizer,
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
    print(f"Saving checkpoint at {path}")
    torch.save(
        {
            "epoch":        epoch,
            "best_val":     best_val,
            "model_state":  model.state_dict(),
            "optim_state":  optim.state_dict(),
            "cat_features": cat_features,
            "cont_features": cont_features,
            "config":       cfg,
        },
        path,
    )



# ──────────────────────────────────────────────────────────────────────────────
#  load / resume
# ──────────────────────────────────────────────────────────────────────────────
def load_ckpt(
    path: str | Path,
) -> Tuple[nn.Module, float, int]:
    """
    Returns
    -------
    model: Model found at given path
    best_val : float
        Best validation loss stored in checkpoint; inf if starting from scratch.
    start_epoch : int
        Epoch index to continue with (0-based). 0 means train from scratch.
    """
    if type(path) == str:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Told model to resume training from a checkpoint, but no checkpoint exists at the given directory: {path}")

    ckpt = torch.load(path, weights_only=False)
    model = TransactionModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])

    optim  = torch.optim.Adam(model.parameters())
    optim.load_state_dict(ckpt["optim_state"])

    best_val = ckpt.get("best_val", float("inf"))
    start_ep = ckpt.get("epoch", 0) + 1

    return model, best_val, start_ep


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
