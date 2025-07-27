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

from config import ModelConfig, LSTMConfig, MLPConfig   # only for typing / pretty storage

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#  save
# ------------------------------------------------------------------------------
def save_ckpt(
    model: nn.Module,
    optim: optim.Optimizer,
    epoch: int,
    best_val: float,
    path: str | Path,
    cat_features: List[str],
    cont_features: List[str],
    cfg: ModelConfig | LSTMConfig | MLPConfig,
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



# ------------------------------------------------------------------------------
#  load / resume
# ------------------------------------------------------------------------------
def load_ckpt(
    path: str | Path,
    device,
) -> Tuple[TransactionModel, float, int]:
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

    if not path.exists(): # type: ignore
        raise FileNotFoundError(f"Told model to resume training from a checkpoint, but no checkpoint exists at the given directory: {path}")
    
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = TransactionModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    best_val = ckpt.get("best_val", float("inf"))
    start_ep = ckpt.get("epoch", 0) + 1
    return model, best_val, start_ep, 

def resume_finetune(
    path: Path,
    device: torch.device,
    unfreeze_backbone: bool = True
) -> Tuple[TransactionModel, float, int, torch.optim.AdamW]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TransactionModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)

    if not unfreeze_backbone:
        for n,p in model.named_parameters():
            if not n.startswith("lstm_head"):
                p.requires_grad = False
            # elif not n.startswith("mlp"):
            #     p.requires_grad=False
    for n, p in model.named_parameters():
        print(n, p.requires_grad)
    # build AdamW over only the grad‑true params (1 group)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3  # dummy - will be overwritten by load_state_dict
    )
    optim.load_state_dict( ckpt["optim_state"] )

    best_val  = ckpt.get("best_val", float("inf"))
    start_ep  = ckpt.get("epoch", 0) + 1
    return model, best_val, start_ep, optim



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



# from collections import Counter

# def count_txn_labels_direct(dataset):
#     counts = Counter()
#     for i in range(len(dataset)):
#         label = int(dataset[i]["label"].item())
#         counts[label] += 1
#     print(f"non fraud: {counts.get(0,0):,d}")
#     print(f"fraud:     {counts.get(1,0):,d}")
#     return counts

# print(count_txn_labels_direct(train_ds))
# print(count_txn_labels_direct(val_ds))


# from collections import Counter
# from torch.utils.data import DataLoader

# def count_txn_labels(dataset, batch_size=1024, num_workers=0):
#     """
#     Iterate through TxnDataset (or via a DataLoader) and count how many
#     windows are labeled fraud (1) vs non fraud (0).
#     """
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn  # or your custom collate_fn
#     )
#     counts = Counter()
#     for batch in loader:
#         labels = batch["label"].flatten().tolist()
#         counts.update(labels)
#     print(f"non fraud: {counts.get(0,0):,d}")
#     print(f"fraud:     {counts.get(1,0):,d}")
#     return counts
# print("dataloader method counting")
# counts = count_txn_labels(train_ds)
# n_neg = counts.get(0, 0)
# n_pos = counts.get(1, 0)
# pos_weight = n_neg / n_pos
# pos_weight = 82.4213860812

# print(f"negatives = {n_neg:,}, positives = {n_pos:,}, pos_weight = {pos_weight:.3f}")
# Example usage:
# dataset = TxnDataset(df, "user_id", cat_feats, cont_feats, window=20, stride=5)
# count_txn_labels(dataset)