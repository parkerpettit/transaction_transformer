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
        Accuracy per categorical feature (0-1 range).
    """
    model.eval()

    total_loss, total_samples = 0.0, 0
    feat_correct = [0] * len(cat_features)
    feat_total   = [0] * len(cat_features)

    sizes = [vocab_sizes[f] for f in cat_features]   # lens in the same order
    for batch in loader:
        # slice_batch = the helper you already have
        inp_cat, inp_cont, tgt_cat, tgt_cont, _ = slice_batch(batch)
        inp_cat, inp_cont = inp_cat.to(device), inp_cont.to(device)
        tgt_cat, tgt_cont = tgt_cat.to(device), tgt_cont.to(device)

        logits_cat, pred_cont = model(inp_cat, inp_cont, mode="ar")

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

        loss_cont = crit_cont(pred_cont, tgt_cont)
        loss = (loss_cat + loss_cont) / (len(vocab_sizes) + 1 )

        
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
Validation helper for the fraud-classification fine-tuning phase.
Returns loss plus a rich metrics dict and takes care of W&B logging:
  • overall accuracy, precision, recall, F1
  • ROC-AUC, PR-AUC
  • class-wise accuracy
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

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate_recall_at_fprs(y_true, y_pred_proba, fpr_thresholds, granularity=2):
    """
    For each fpr_limit in fpr_thresholds, find the threshold that gives the highest recall
    while keeping FPR <= fpr_limit.  If the only way to get FPR <= fpr_limit is to predict
    no positives (threshold=1.0), fall back to the threshold (≠1.0) whose FPR is closest
    to fpr_limit.
    """
    results = []
    step = 10 * (0.1 ** granularity)

    for fpr_limit in fpr_thresholds:
        best_recall = 0.0
        best_thr = 1.0

        # --- 1) binary search for threshold satisfying FPR <= fpr_limit ---
        low, high = 0.0, 1.0
        while abs(high - low) >= step:
            mid = round((low + high) / 2, granularity)
            preds = (y_pred_proba >= mid).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            fpr    = fp / (fp + tn) if (fp + tn) else 0.0

            # if under the limit, consider this threshold for best recall
            if fpr <= fpr_limit and recall > best_recall:
                best_recall = recall
                best_thr    = mid

            # narrow search window
            if fpr > fpr_limit:
                low = mid
            else:
                high = mid

        # --- 2) fallback: if best_thr == 1.0 (i.e. trivial) do a brute‐force pass ---
        if best_recall == 0.0 and best_thr == 1.0:
            thrs = np.unique(y_pred_proba)
            fprs = []
            recs = []
            for thr in thrs:
                preds = (y_pred_proba >= thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
                fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
                recs.append(tp / (tp + fn) if (tp + fn) else 0.0)
            fprs = np.array(fprs)
            recs = np.array(recs)

            # ignore the trivial threshold=1.0
            mask = thrs < 1.0
            # pick the threshold whose FPR is closest to the target
            idx = np.argmin(np.abs(fprs[mask] - fpr_limit))
            best_thr    = thrs[mask][idx]
            best_recall = recs[mask][idx]

        results.append({
            'fpr_limit':   fpr_limit,
            'threshold':   best_thr,
            'best_recall': best_recall
        })

    return pd.DataFrame(results)

from typing import List, Dict, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

@torch.no_grad()
def evaluate_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.BCEWithLogitsLoss,
    device: torch.device,
    mode: str,
    threshold: float = 0.5,
    class_names: List[str] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a binary (fraud vs non-fraud) model using BCEWithLogitsLoss.
    Logs scalar metrics, ROC/PR curves, and confusion matrices at various FPR thresholds.
    """
    model.eval()
    tot_loss = 0.0
    tot_samples = 0

    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in loader:
        # Unpack and move to device
        cat = batch["cat"][:, :-1].to(device)
        con = batch["cont"][:, :-1].to(device)
        labels = batch["label"].to(device).float()
        labels_int = labels.long()

        # Forward + loss
        logits = model(cat, con, mode=mode)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        # Accumulate loss & counts
        bs = labels_int.size(0)
        tot_loss   += loss.item() * bs
        tot_samples += bs

        all_probs .extend(probs.cpu().tolist())
        all_labels.extend(labels_int.cpu().tolist())

        # Per-class accuracy
    

    # Compute aggregated metrics
    val_loss = tot_loss / tot_samples
    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)
    probas_2d = np.vstack([1 - probs_np, probs_np]).T

    fpr_thresholds = [0.01, 0.001, 0.0005, 0.0001]
    df_thr = evaluate_recall_at_fprs(labels_np, probs_np, fpr_thresholds)
    # extract threshold for 0.001 as
    threshold = df_thr.loc[df_thr.fpr_limit==0.01, "threshold"].item()
        # now binarize at that threshold
    preds_np = (probs_np >= threshold).astype(int)
    # Confusion matrices at targeted FPR thresholds
    class_labels = class_names or ["non-fraud", "fraud"]

    for _, row in df_thr.iterrows():
        thr = row["threshold"]
        fpr_lim = row["fpr_limit"]
        y_pred_thr = (probs_np >= thr).astype(int)
        cm = confusion_matrix(labels_np, y_pred_thr, labels=[0,1])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set(
        title=f"CM @ FPR≤{fpr_lim*100:.2f}% (thr={thr:.4f})"
        )
        key = f"confusion_matrix_fpr_{fpr_lim*100:.2f}pct"
        wandb.log({key: wandb.Image(fig)}, commit=False)
    
    accuracy = (preds_np == labels_np).mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division=0
    )
    class_acc = {
      0: (preds_np[labels_np==0]==0).mean(),
      1: (preds_np[labels_np==1]==1).mean(),
    }
    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = float("nan")

    # Log scalar metrics

    # Log ROC & PR curves
    roc_plot = wandb.plot.roc_curve(
        y_true   = labels_np.tolist(),
        y_probas = probas_2d.tolist(),
        labels   = class_labels,
    )
    pr_plot = wandb.plot.pr_curve(
        y_true   = labels_np.tolist(),
        y_probas = probas_2d.tolist(),
        labels   = class_labels,
    )
    wandb.log({"roc_curve": roc_plot, "pr_curve": pr_plot}, commit=False)


    wandb.log({
        "val_loss":       val_loss,
        "val_accuracy":   accuracy,
        "val_precision":  precision,
        "val_recall":     recall,
        "val_f1":         f1,
        "val_roc_auc":    roc_auc,
        "val_pr_auc":     pr_auc,
        "val_non_fraud_acc": class_acc[0],
        "val_fraud_acc": class_acc[1],  
        }, commit=True)
    model.train()
    plt.close("all")
    return val_loss, {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "roc_auc":     roc_auc,
        "pr_auc":      pr_auc,
        "class_acc_0": class_acc[0],
        "class_acc_1": class_acc[1],
    } # type: ignore
