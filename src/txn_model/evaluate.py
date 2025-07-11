import logging
import torch
import torch.nn as nn
from config import (
    ModelConfig,
    FieldTransformerConfig,
    SequenceTransformerConfig,
    LSTMConfig,
)
from model import TransactionModel

from data.dataset import slice_batch

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cat_features: List[str],          # names in the same order you sliced tgt_cat
    vocab_sizes: Dict[str, int],   # {feature_name: vocab_len}
    crit_cat: nn.Module,           # e.g. nn.CrossEntropyLoss()
    crit_cont: nn.Module,          # e.g. nn.MSELoss()
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate once over `loader`.

    Returns
    -------
    avg_loss : float
        Mean loss (categorical + continuous) over all samples.
    feat_acc : dict[str, float]
        Accuracy per categorical feature (0–1 range).
    """
    model.eval()

    total_loss, total_samples = 0.0, 0
    feat_correct = [0] * len(cat_features)
    feat_total   = [0] * len(cat_features)

    sizes = [vocab_sizes[f] for f in cat_features]   # lens in the same order
    for batch in loader:
        # slice_batch = the helper you already have
        inp_cat, inp_cont, inp_mask, tgt_cat, tgt_cont = slice_batch(batch)
        inp_cat, inp_cont = inp_cat.to(device), inp_cont.to(device)
        inp_mask          = inp_mask.to(device).bool()
        tgt_cat, tgt_cont = tgt_cat.to(device), tgt_cont.to(device)

        logits_cat, pred_cont = model(inp_cat, inp_cont, inp_mask, mode="ar")

        # ----- compute loss & per-feature accuracy -----
        start, loss_cat = 0, 0.0
        for i, V in enumerate(sizes):
            end        = start + V
            logits_f   = logits_cat[:, start:end]        # [B, V_i]
            targets_f  = tgt_cat[:, i]                   # [B]
            loss_cat  += crit_cat(logits_f, targets_f)

            preds_f     = logits_f.argmax(dim=1)
            feat_correct[i] += (preds_f == targets_f).sum().item()
            feat_total[i]   += targets_f.numel()
            start = end

        loss = loss_cat + crit_cont(pred_cont, tgt_cont)
        batch_size   = inp_cat.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    feat_acc = {
        name: (feat_correct[i] / feat_total[i]) if feat_total[i] else 0.0
        for i, name in enumerate(cat_features)
    }

    model.train()           # restore training mode
    return avg_loss, feat_acc






"""
evaluate_binary.py
------------------
Validation for the fraud-classification phase.
Returns aggregate loss, overall accuracy, and per-class accuracy.
"""

from typing import Dict, Tuple

import torch
from torch import nn


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Parameters
    ----------
    model      : TransactionModel in 'fraud' mode
    loader     : validation DataLoader
    criterion  : nn.CrossEntropyLoss (or BCEWithLogitsLoss)
    device     : torch.device

    Returns
    -------
    val_loss   : float
    val_acc    : float  (overall)
    class_acc  : {class_id: accuracy}
    """
    model.eval()

    tot_loss, tot_correct, tot_samples = 0.0, 0, 0
    cls_correct: Dict[int, int] = {}
    cls_total:   Dict[int, int] = {}

    for batch in loader:
        cat = batch["cat"][:, :-1].to(device)
        con = batch["cont"][:, :-1].to(device)
        pad = batch["pad_mask"][:, :-1].bool().to(device)
        y   = batch["label"].to(device)

        logits = model(cat, con, pad, mode="fraud")

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            logits = logits.view(-1)
            loss   = criterion(logits, y.float())
            preds  = (torch.sigmoid(logits) > 0.5).long()
        else:                                      # CrossEntropy
            loss   = criterion(logits, y)
            preds  = logits.argmax(dim=1)

        batch_size = y.size(0)
        tot_loss   += loss.item() * batch_size
        tot_correct += (preds == y).sum().item()
        tot_samples += batch_size

        for cls_id in y.unique().tolist():
            mask = y == cls_id
            cls_correct[cls_id] = cls_correct.get(cls_id, 0) + (preds[mask] == y[mask]).sum().item()
            cls_total[cls_id]   = cls_total.get(cls_id, 0)   + mask.sum().item()

    model.train()

    val_loss = tot_loss / tot_samples
    val_acc  = tot_correct / tot_samples
    class_acc = {c: cls_correct[c] / cls_total[c] for c in cls_total}

    return val_loss, val_acc, class_acc