from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

import wandb
from data.dataset import slice_batch

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
    print(f"  └- Avg Val Loss: {avg_loss:.4f}\n")

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

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import wandb
import torch

# ---------- Helper: exact thresholds for FPR limits ----------
def thresholds_for_fpr_limits(y_true: np.ndarray,
                              y_score: np.ndarray,
                              fpr_limits: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_limits: one row per requested fpr_limit with chosen threshold and metrics.
      curve: full monotone curve at every unique probability (optional for analysis).
    """
    y_true  = np.asarray(y_true,  dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)

    order = np.argsort(-y_score)               # descending scores
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    # block ends (unique thresholds)
    change = np.empty_like(scores_sorted, dtype=bool)
    change[:-1] = scores_sorted[:-1] != scores_sorted[1:]
    change[-1] = True
    block_idx = np.nonzero(change)[0]

    pos_mask = (y_sorted == 1)
    cum_pos = np.cumsum(pos_mask)
    cum_neg = np.cumsum(~pos_mask)

    total_pos = int(cum_pos[-1])
    total_neg = int(cum_neg[-1])

    tp = cum_pos[block_idx]
    fp = cum_neg[block_idx]
    fn = total_pos - tp
    tn = total_neg - fp

    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    fpr = fp / total_neg if total_neg else np.zeros_like(fp, dtype=float)
    thresholds = scores_sorted[block_idx]

    # fpr is non‑decreasing w.r.t. lowering threshold
    df_curve = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    })

    results = []
    for limit in fpr_limits:
        idx = np.searchsorted(fpr, limit, side="right") - 1
        if idx >= 0 and fpr[idx] <= limit:
            chosen = idx
        else:
            # no feasible point (other than predicting none) under limit:
            # choose closest; prefer undershoot
            diffs = np.abs(fpr - limit)
            penalty = (fpr > limit).astype(int)
            # lexicographic: (penalty, diff, threshold) -> pick first
            chosen = np.lexsort((thresholds, diffs, penalty))[0]

        results.append({
            "fpr_limit": float(limit),
            "threshold": float(thresholds[chosen]),
            "fpr": float(fpr[chosen]),
            "recall": float(recall[chosen]),
            "tp": int(tp[chosen]),
            "fp": int(fp[chosen]),
            "fn": int(fn[chosen]),
            "tn": int(tn[chosen]),
        })

    return pd.DataFrame(results), df_curve


@torch.no_grad()
def evaluate_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.BCEWithLogitsLoss,
    device: torch.device,
    mode: str,
    class_names: list[str] | None = None,
):
    model.eval()

    tot_loss = 0.0
    tot_samples = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for batch in loader:
        cat = batch["cat"][:, :-1].to(device)
        con = batch["cont"][:, :-1].to(device)
        labels = batch["label"].to(device).float()
        labels_int = labels.long()

        logits = model(cat, con, mode=mode)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        bs = labels_int.size(0)
        tot_loss += loss.item() * bs
        tot_samples += bs

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels_int.cpu().tolist())

    val_loss = tot_loss / tot_samples
    labels_np = np.asarray(all_labels, dtype=np.int8)
    probs_np  = np.asarray(all_probs,  dtype=np.float32)
    probas_2d = np.vstack([1 - probs_np, probs_np]).T

    # ---- FPR-constrained thresholds ----
    fpr_limits = [0.01, 0.001, 0.0005, 0.0001]
    df_limits, _ = thresholds_for_fpr_limits(labels_np, probs_np, fpr_limits)

    # Choose one threshold to compute "main" metrics (example: use 1% FPR row)
    main_thr = df_limits.loc[df_limits.fpr_limit == 0.01, "threshold"].item()
    preds_np = (probs_np >= main_thr).astype(int)

    class_labels = class_names or ["non-fraud", "fraud"]

    # Log confusion matrices at each FPR limit with actual achieved FPR
    for _, row in df_limits.iterrows():
        thr = row.threshold
        fpr_lim = row.fpr_limit
        got_fpr = row.fpr
        y_pred_thr = (probs_np >= thr).astype(int)
        cm = confusion_matrix(labels_np, y_pred_thr, labels=[0,1])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set(title=f"CM @ target FPR≤{fpr_lim*100:.2f}% (thr={thr:.6f})\nAchieved FPR={got_fpr*100:.4f}%, Recall={row.recall*100:.4f}%")
        key = f"confusion_matrix_fpr_{fpr_lim*100:.2f}pct"
        wandb.log({key: wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # Scalar metrics at main_thr
    accuracy = (preds_np == labels_np).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division=0
    )
    class_acc = {
        0: (preds_np[labels_np == 0] == 0).mean(),
        1: (preds_np[labels_np == 1] == 1).mean(),
    }
    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = float("nan")

    # ROC / PR curves
    roc_plot = wandb.plot.roc_curve(
        y_true=labels_np.tolist(),
        y_probas=probas_2d.tolist(),
        labels=class_labels,
    )
    pr_plot = wandb.plot.pr_curve(
        y_true=labels_np.tolist(),
        y_probas=probas_2d.tolist(),
        labels=class_labels,
    )
    wandb.log({"roc_curve": roc_plot, "pr_curve": pr_plot}, commit=False)

    # Log scalar metrics + chosen threshold metrics + per-limit table
    # Flatten df_limits rows into a dict (optional)
    limit_logs = {}
    for _, r in df_limits.iterrows():
        tag = f"fpr_{r.fpr_limit:.6f}"
        limit_logs[f"{tag}_thr"] = r.threshold
        limit_logs[f"{tag}_achieved_fpr"] = r.fpr
        limit_logs[f"{tag}_recall"] = r.recall
        limit_logs[f"{tag}_tp"] = r.tp
        limit_logs[f"{tag}_fp"] = r.fp
        limit_logs[f"{tag}_fn"] = r.fn
        limit_logs[f"{tag}_tn"] = r.tn

    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_roc_auc": roc_auc,
        "val_pr_auc": pr_auc,
        "val_non_fraud_acc": class_acc[0],
        "val_fraud_acc": class_acc[1],
        "main_threshold_fpr01": main_thr,
        **limit_logs
    }, commit=True)

    model.train()
    return val_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "class_acc_0": class_acc[0],
        "class_acc_1": class_acc[1],
        "thresholds_table": df_limits 
    }
