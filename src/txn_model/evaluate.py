import logging
import torch
import torch.nn as nn
from config import (
    ModelConfig,
    TransformerConfig,
    LSTMConfig,
)
from model import TransactionModel

from data.dataset import slice_batch

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch
from torch import nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    average_precision_score, matthews_corrcoef
)
import wandb
import sklearn


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
        loss_cat /= len(vocab_sizes)
        loss_cont = crit_cont(pred_cont, tgt_cont)
        loss = loss_cat + loss_cont 
        
        batch_size   = inp_cat.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    feat_acc = {
        name: (feat_correct[i] / feat_total[i]) if feat_total[i] else 0.0
        for i, name in enumerate(cat_features)
    }

    print("\n▶ Feature-wise Accuracy:")
    for name, acc in feat_acc.items():
        print(f"  - {name:<20}: {acc*100:.2f}%")
    print(f"  └─ Avg Val Loss: {avg_loss:.4f}\n")

    model.train()           # restore training mode
    return avg_loss, feat_acc








"""
evaluate_binary.py
------------------
Validation helper for the fraud‑classification fine‑tuning phase.
Returns loss plus a rich metrics dict and takes care of W&B logging:
  • overall accuracy, precision, recall, F1
  • ROC‑AUC, PR‑AUC
  • class‑wise accuracy
  • confusion matrix, ROC curve, PR curve as interactive W&B plots
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import wandb


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    class_names: List[str] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Run evaluation and log metrics to Weights & Biases.

    Parameters
    ----------
    model      : ``TransactionModel`` already switched to ``mode="fraud"``
    loader     : validation ``DataLoader``
    criterion  : ``nn.BCEWithLogitsLoss`` (preferred) *or* ``nn.CrossEntropyLoss``
    device     : CUDA / CPU device handle
    threshold  : Probability cut-off for turning scores into 0/1 labels (default 0.5)
    class_names: Optional list like ``["non-fraud", "fraud"]``

    Returns
    -------
    val_loss : float - mean loss over the loader
    metrics  : Dict[str, float] - keys include ``acc, precision, recall, f1, roc_auc, pr_auc``
    """

    model.eval()

    tot_loss: float = 0.0
    tot_samples: int = 0

    # Accumulate for global metrics
    all_probs:  List[float] = []
    all_preds:  List[int]   = []
    all_labels: List[int]   = []

    # Track per‑class accuracy (0 / 1)
    cls_correct: Dict[float, int] = {0: 0, 1: 0}
    cls_total:   Dict[float, int] = {0: 0, 1: 0}
    

    for batch in loader:
        # val_fraud, val_nonfraud = 0, 0
        cat = batch["cat"][:, :-1].to(device)
        con = batch["cont"][:, :-1].to(device)
        pad = batch["pad_mask"][:, :-1].bool().to(device)
        y   = batch["label"].to(device).float()
        uniques, counts = torch.unique(y, return_counts=True)

        # print them
        for value, count in zip(uniques.tolist(), counts.tolist()):
            print(f"{value}: {count}")
        # val_fraud     += y.sum().item()
        # val_nonfraud  += y.size(0) - y.sum().item()
        # print(
        #     f"Validation set — fraud: {val_fraud:,d}, "
        #     f"non-fraud: {val_nonfraud:,d}"
        # )

        logits = model(cat, con, pad, mode="fraud")  # (B,1) or (B,2)

    
        logits = logits.squeeze(1)          # (B,)
        loss   = criterion(logits, y.float())
        probs  = torch.sigmoid(logits)
        

        preds = (probs > 0.5).float()

        # Book‑keeping
        batch_size = y.size(0)
        tot_loss   += loss.item() * batch_size
        tot_samples += batch_size

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        for c in [0.0, 1.0]:
            mask = (y == c)
            cls_total[c]   += mask.sum().item()
            cls_correct[c] += (preds[mask] == y[mask]).sum().item()

    # Aggregate metrics
    val_loss = tot_loss / tot_samples
    labels_np = np.array(all_labels)
    preds_np  = np.array(all_preds)
    probs_np  = np.array(all_probs)
    # probs_np is shape (n,)  —  probability of class “fraud”
    # build a (n,2) array: [P(non-fraud), P(fraud)] for each sample
    probas_2d = np.vstack([1 - probs_np, probs_np]).T   # shape (n,2)

    roc_plot = wandb.plot.roc_curve(
        y_true   = labels_np.tolist(),
        y_probas = probas_2d.tolist(),      # <-- now a list of [p0, p1]
        labels   = ["non-fraud", "fraud"],
    )
    acc       = (preds_np == labels_np).mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division=0
    )

    # AUCs can fail if only one class present; guard against that
    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = float("nan")

    class_acc = {
        0: cls_correct[0] / max(cls_total[0], 1),
        1: cls_correct[1] / max(cls_total[1], 1),
    }

    # Log numeric metrics
    wandb.log({
        "val_loss":      val_loss,
        "val_accuracy":  acc,
        "val_precision": precision,
        "val_recall":    recall,
        "val_f1":        f1,
        "val_roc_auc":   roc_auc,
        "val_pr_auc":    pr_auc,
        "val_class_acc_0": class_acc[0],
        "val_class_acc_1": class_acc[1],
    })

    # Log plots (confusion matrix, ROC, PR)
    class_labels = class_names or ["non-fraud", "fraud"]
    cm_plot = wandb.sklearn.plot_confusion_matrix( # type: ignore
        labels_np.tolist(),
        preds_np.tolist(),
        class_labels,
    )

    pr_plot = wandb.plot.pr_curve(
        y_true   = labels_np.tolist(),
        y_probas = probas_2d.tolist(),
        labels   = class_labels,
    )
    wandb.log({
        "confusion_matrix": cm_plot,
        "roc_curve":        roc_plot,
        "pr_curve":         pr_plot,
    })

    metrics = {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
        "class_acc_0": class_acc[0],
        "class_acc_1": class_acc[1],
    }
    from collections import Counter

    pred_counts  = Counter(all_preds)
    label_counts = Counter(all_labels)

    print(
        f"PREDICTION COUNTS → "
        f"non-fraud (0): {pred_counts.get(0,0):,d}, "
        f"fraud (1): {pred_counts.get(1,0):,d}"
    )
    print(
        f"LABEL COUNTS → "
        f"non-fraud (0): {label_counts.get(0,0):,d}, "
        f"fraud (1): {label_counts.get(1,0):,d}"
    )

    model.train()
    return val_loss, metrics
