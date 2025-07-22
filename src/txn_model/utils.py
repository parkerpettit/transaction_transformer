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
from typing import List, Dict, Tuple

from config import ModelConfig   # only for typing / pretty storage

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
    model.load_state_dict(ckpt["model_state"])

    if not unfreeze_backbone:
        for n,p in model.named_parameters():
            if not n.startswith("lstm_head"):
                p.requires_grad = False
            elif not n.startswith("mlp"):
                p.requires_grad=False
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


import torch

def show_samples_mlm(
    inp_cat: torch.Tensor,      # [B, L, F_cat] masked inputs
    logits_cat: torch.Tensor,   # [B, L, ΣVᵢ] raw logits over all cat features
    mask_cat: torch.Tensor,     # [B, L, F_cat] bool mask of which cat slots were masked
    tgt_cat: torch.Tensor,      # [B, L, F_cat] original cat labels (before masking)
    inp_cont: torch.Tensor,     # [B, L, F_cont] masked cont inputs
    pred_cont: torch.Tensor,    # [B, L, F_cont] model’s cont predictions
    mask_cont: torch.Tensor,    # [B, L, F_cont] bool mask of which cont slots were masked
    tgt_cont: torch.Tensor,     # [B, L, F_cont] original cont labels (before masking)
    cat_features: List[str],
    cont_features: List[str],
    enc: Dict[str, Dict[str, list]],
    n: int = 3
):
    """
    Print up to `n` samples' predictions vs. targets for all masked slots.

    Args:
      inp_cat     Tensor[B,L,F_cat]    - masked categorical inputs
      logits_cat  Tensor[B,L,ΣVᵢ]      - raw logits for all cat features
      mask_cat    BoolTensor[B,L,F_cat]- which cat slots were masked
      tgt_cat     Tensor[B,L,F_cat]    - original cat values
      inp_cont    Tensor[B,L,F_cont]   - masked continuous inputs
      pred_cont   Tensor[B,L,F_cont]   - cont predictions
      mask_cont   BoolTensor[B,L,F_cont]- which cont slots were masked
      tgt_cont    Tensor[B,L,F_cont]   - original cont values
      cat_features List[str]           - names of your cat fields
      cont_features List[str]          - names of your cont fields
      enc          dict                - enc[f]["inv"] maps code→string
      n            int                 - how many batch samples to show
    """
    B, L, _ = logits_cat.shape
    F_cat = len(cat_features)
    F_cont = len(cont_features)

    # 1) split logits_cat into per‐feature preds
    sizes = [len(enc[f]["inv"]) for f in cat_features]
    cat_preds = torch.zeros((B, L, F_cat), dtype=torch.long, device=logits_cat.device)
    start = 0
    for i, size in enumerate(sizes):
        slice_logits = logits_cat[:, :, start:start+size]  # [B, L, size]
        cat_preds[:, :, i] = slice_logits.argmax(dim=-1)
        start += size

    # 2) decoding helper
    def decode_cat(code: int, feat: str):
        inv = enc[feat]["inv"]
        return inv[code] if 0 <= code < len(inv) else f"<UNK:{code}>"

    n = min(n, B)
    for b in range(n):
        print(f"\n─ Sample {b} ─")
        for t in range(L):
            # categorical fields
            for i, feat in enumerate(cat_features):
                if mask_cat[b, t, i]:
                    tgt = tgt_cat[b, t, i].item()
                    pred = cat_preds[b, t, i].item()
                    print(f"[t={t:>2}] {feat:<18}"
                          f" tgt={decode_cat(tgt, feat):<12}" # type: ignore
                          f" pred={decode_cat(pred, feat)}") # type: ignore
            # continuous fields
            for j, feat in enumerate(cont_features):
                if mask_cont[b, t, j]:
                    tgt = tgt_cont[b, t, j].item()
                    pred = pred_cont[b, t, j].item()
                    print(f"[t={t:>2}] {feat:<18}"
                          f" tgt={tgt:>8.3f}"
                          f" pred={pred:>8.3f}")
        print("─" * 40)
