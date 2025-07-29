#!/usr/bin/env python
# Comprehensive LightGBM baseline with exhaustive threshold analysis & metrics
# - Finds global max-F1 threshold
# - Finds max-F1 thresholds subject to given FPR (false positive rate) limits
# - Computes & logs a wide range of metrics for each selected threshold
# - Plots confusion matrices annotated with key metrics (F1 / Precision / Recall / FPR / Threshold)
# - Logs ROC, PR curves, and a thresholds table
# - Single pass over probability scores for efficiency

import argparse
import random
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# --------------------------------------------------------------------------- #
# Utility: reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --------------------------------------------------------------------------- #
# Core: build threshold curve once (unique probability cutpoints)
# --------------------------------------------------------------------------- #
def build_threshold_curve(y_true, y_prob):
    """
    Returns per-unique-threshold arrays:
      thresholds, tp, fp, tn, fn, prec, recall, fpr, tnr, fnr, npv, fdr, forate, f1, f2, f05, mcc
    Monotonic: as index increases, threshold decreases or stays same (scores sorted desc).
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    order = np.argsort(-y_prob)  # descending score
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    total_pos = int((y_true == 1).sum())
    total_neg = int((y_true == 0).sum())

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)

    # first occurrence of each unique prob while descending
    unique_idx = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1]])

    tp = cum_tp[unique_idx].astype(np.int64)
    fp = cum_fp[unique_idx].astype(np.int64)
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = p_sorted[unique_idx]

    # Avoid div-by-zero with np.where
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp / np.where(tp + fn > 0, tp + fn, 1)         # TPR
        precision = tp / np.where(tp + fp > 0, tp + fp, 1)
        fpr = fp / np.where(fp + tn > 0, fp + tn, 1)            # FPR = 1 - TNR
        tnr = tn / np.where(tn + fp > 0, tn + fp, 1)            # specificity
        fnr = fn / np.where(fn + tp > 0, fn + tp, 1)
        npv = tn / np.where(tn + fn > 0, tn + fn, 1)
        fdr = fp / np.where(tp + fp > 0, tp + fp, 1)
        forate = fn / np.where(fn + tn > 0, fn + tn, 1)         # false omission rate
        f1 = 2 * precision * recall / np.where(precision + recall > 0, precision + recall, 1)
        beta2 = 2.0
        f2 = (1 + beta2**2) * precision * recall / np.where(beta2**2 * precision + recall > 0,
                                                            beta2**2 * precision + recall, 1)
        beta05 = 0.5
        f05 = (1 + beta05**2) * precision * recall / np.where(beta05**2 * precision + recall > 0,
                                                              beta05**2 * precision + recall, 1)
        # MCC
        denom = np.sqrt(
            np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
        )
        mcc = (tp * tn - fp * fn) / denom

    return {
        "thresholds": thresholds,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "npv": npv,
        "fdr": fdr,
        "for": forate,
        "f1": f1,
        "f2": f2,
        "f05": f05,
        "mcc": mcc,
    }


# --------------------------------------------------------------------------- #
# Select thresholds
# --------------------------------------------------------------------------- #
def select_global_max_f1(curve):
    """
    Returns dict with stats at threshold maximizing F1.
    Tie-break: higher F1, then higher recall, then lower threshold.
    """
    f1 = curve["f1"]
    recall = curve["recall"]
    thr = curve["thresholds"]
    # Build tie-break key (negative for max)
    keys = np.lexsort((thr, -recall, -f1))  # lexsort uses last array as primary
    # We sorted keys giving ascending of each key; we want max f1 so invert logic:
    # Simple: take index of global max f1, then among ties pick higher recall then lower threshold.
    max_f1 = f1.max()
    cand = np.where(f1 == max_f1)[0]
    # Among candidates choose one with highest recall then lowest threshold
    sub_rec = recall[cand]
    best_local = cand[np.lexsort((thr[cand], -sub_rec))][0]
    return extract_row(curve, best_local, label="global_max_f1")


def select_max_f1_under_fpr(curve, target_fpr):
    """
    Maximize F1 subject to fpr <= target_fpr.
    If no threshold satisfies, pick closest fpr (undershoot preferred; tie -> smaller abs diff, then lower threshold).
    """
    fpr = curve["fpr"]
    feasible = np.where(fpr <= target_fpr)[0]
    if feasible.size > 0:
        f1 = curve["f1"][feasible]
        recall = curve["recall"][feasible]
        thr = curve["thresholds"][feasible]
        max_f1 = f1.max()
        cand = feasible[np.where(f1 == max_f1)[0]]
        # Among equal F1 pick higher recall then lower threshold
        sub_rec = curve["recall"][cand]
        chosen = cand[np.lexsort((curve["thresholds"][cand], -sub_rec))][0]
        return extract_row(curve, chosen, label=f"max_f1_fpr_le_{target_fpr}")
    # fallback
    diffs = np.abs(fpr - target_fpr)
    penalty = (fpr > target_fpr).astype(int)  # prefer undershoot (0)
    thr = curve["thresholds"]
    idx = np.lexsort((thr, diffs, penalty))[0]
    row = extract_row(curve, idx, label=f"closest_fpr_{target_fpr}_fallback")
    row["fallback"] = True
    return row


def extract_row(curve, idx, label):
    tp = int(curve["tp"][idx]); fp = int(curve["fp"][idx])
    tn = int(curve["tn"][idx]); fn = int(curve["fn"][idx])
    precision = float(curve["precision"][idx])
    recall = float(curve["recall"][idx])
    fpr = float(curve["fpr"][idx])
    tnr = float(curve["tnr"][idx])
    fnr = float(curve["fnr"][idx])
    npv = float(curve["npv"][idx])
    fdr = float(curve["fdr"][idx])
    forate = float(curve["for"][idx])
    f1 = float(curve["f1"][idx])
    f2 = float(curve["f2"][idx])
    f05 = float(curve["f05"][idx])
    mcc = float(curve["mcc"][idx])

    support_pos = tp + fn
    support_neg = tn + fp
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_acc = 0.5 * (recall + tnr)
    prevalence = support_pos / max(tp + tn + fp + fn, 1)
    pred_pos_rate = (tp + fp) / max(tp + tn + fp + fn, 1)
    youden_j = recall - fpr
    informedness = youden_j
    markedness = precision + npv - 1

    return {
        "label": label,
        "threshold": float(curve["thresholds"][idx]),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "npv": npv,
        "fdr": fdr,
        "for": forate,
        "f1": f1,
        "f2": f2,
        "f05": f05,
        "mcc": mcc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "prevalence": prevalence,
        "pred_pos_rate": pred_pos_rate,
        "youden_j": youden_j,
        "informedness": informedness,
        "markedness": markedness,
        "support_pos": support_pos,
        "support_neg": support_neg,
        "fallback": False,
    }

# --------------------------------------------------------------------------- #
# Metrics / plotting
# --------------------------------------------------------------------------- #
def log_confusion_matrix_with_metrics(y_true, threshold_stats, probs, key_prefix):
    thr = threshold_stats["threshold"]
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-fraud", "fraud"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(
        f"{threshold_stats['label']}\n"
        f"thr={thr}  F1={threshold_stats['f1']:.6f}  "
        f"Prec={threshold_stats['precision']:.6f}  Rec={threshold_stats['recall']:.6f}  "
        f"FPR={threshold_stats['fpr']:.6f}  MCC={threshold_stats['mcc']:.6f}"
    )
    ax.set_xlabel(
        f"Acc={threshold_stats['accuracy']:.6f}  BalAcc={threshold_stats['balanced_accuracy']:.6f}  "
        f"F2={threshold_stats['f2']:.6f}  F0.5={threshold_stats['f05']:.6f}"
    )
    plt.tight_layout()
    wandb.log({f"{key_prefix}/confusion_matrix": wandb.Image(fig)}, commit=False)
    plt.close(fig)


def log_roc_pr(y_true, probs, class_labels=("non-fraud", "fraud")):
    y_true_list = np.asarray(y_true, dtype=np.int8).tolist()
    probs = np.asarray(probs, dtype=np.float32)
    probas_2d = np.vstack([1 - probs, probs]).T.tolist()
    roc_plot = wandb.plot.roc_curve(y_true=y_true_list, y_probas=probas_2d, labels=class_labels)
    pr_plot = wandb.plot.pr_curve(y_true=y_true_list, y_probas=probas_2d, labels=class_labels)
    wandb.log({"roc_curve": roc_plot, "pr_curve": pr_plot}, commit=False)


# --------------------------------------------------------------------------- #
# Argument Parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", type=str, default="lightgbm_baseline")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_path", type=str, default="data/full_processed.pt")

    # LightGBM params
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=127)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--subsample_freq", type=int, default=0)
    p.add_argument("--feature_fraction", type=float, default=1.0)
    p.add_argument("--min_data_in_leaf", type=int, default=20)
    p.add_argument("--min_sum_hessian_in_leaf", type=float, default=1e-3)
    p.add_argument("--min_gain_to_split", type=float, default=0.0)
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--lambda_l2", type=float, default=0.0)
    p.add_argument("--boosting_type", type=str, default="gbdt", choices=["gbdt", "dart", "goss"])
    p.add_argument("--objective", type=str, default="binary")
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--is_unbalance", action="store_true")

    # Threshold analysis
    p.add_argument(
        "--fpr_limits",
        type=str,
        default="0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001",
        help="Comma-separated target FPR values; for each we find max-F1 under the limit."
    )
    p.add_argument("--save_best_only", action="store_true",
                   help="Save only best iteration booster.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), reinit=False)

    # Load data (expects (train_df, val_df, ...) saved via torch.save)
    train_df, val_df, *_ = torch.load(args.data_path, weights_only=False)
    y_train = train_df["is_fraud"].values
    X_train = train_df.drop(columns="is_fraud")
    y_val = val_df["is_fraud"].values
    X_val = val_df.drop(columns="is_fraud")

    clf = lgb.LGBMClassifier(
        objective=args.objective,
        boosting_type=args.boosting_type,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        feature_fraction=args.feature_fraction,
        min_data_in_leaf=args.min_data_in_leaf,
        min_sum_hessian_in_leaf=args.min_sum_hessian_in_leaf,
        min_gain_to_split=args.min_gain_to_split,
        reg_alpha=args.lambda_l1,
        reg_lambda=args.lambda_l2,
        is_unbalance=args.is_unbalance,
        verbosity=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
    )

    probs = clf.predict_proba(X_val)[:, 1].astype(np.float32)

    # Global metrics (score-based)
    roc_auc = roc_auc_score(y_val, probs)
    pr_auc = average_precision_score(y_val, probs)

    # Threshold curve (single pass)
    curve = build_threshold_curve(y_val, probs)

    # Global max F1 threshold
    max_f1_stats = select_global_max_f1(curve)

    # FPR-limited thresholds
    fpr_limits = []
    if args.fpr_limits.strip():
        for tk in args.fpr_limits.split(","):
            tk = tk.strip()
            if tk:
                try:
                    v = float(tk)
                    if v >= 0:
                        fpr_limits.append(v)
                except ValueError:
                    pass
    fpr_limits = sorted(set(fpr_limits))
    fpr_threshold_stats = [select_max_f1_under_fpr(curve, lim) for lim in fpr_limits]

    # Build a full table of all chosen thresholds
    rows = [max_f1_stats] + fpr_threshold_stats
    table_cols = [
        "label", "threshold",
        "f1", "f2", "f05",
        "precision", "recall",
        "accuracy", "balanced_accuracy",
        "fpr", "tnr", "fnr",
        "mcc",
        "npv", "fdr", "for",
        "youden_j", "informedness", "markedness",
        "pred_pos_rate", "prevalence",
        "tp", "fp", "tn", "fn",
        "support_pos", "support_neg",
        "fallback"
    ]
    wb_table = wandb.Table(columns=table_cols)
    for r in rows:
        wb_table.add_data(*[r[c] for c in table_cols])
    wandb.log({"val/selected_thresholds_table": wb_table}, commit=False)

    # Log confusion matrices
    log_confusion_matrix_with_metrics(y_val, max_f1_stats, probs, "val/max_f1")
    for r in fpr_threshold_stats:
        prefix = f"val/fpr_{r['fpr']:.10f}"
        log_confusion_matrix_with_metrics(y_val, r, probs, prefix)

    # Also baseline threshold 0.5 (if not already captured)
    baseline_stats = stats_at_fixed_threshold(y_val, probs, 0.5)
    log_confusion_matrix_with_metrics(y_val, baseline_stats, probs, "val/threshold_0_5")

    # Log ROC/PR interactive
    log_roc_pr(y_val, probs)

    # Scalar summary logs (global & key thresholds)
    wandb.log({
        "val/roc_auc": roc_auc,
        "val/average_precision": pr_auc,
        # Global max F1
        "val/max_f1_threshold": max_f1_stats["threshold"],
        "val/max_f1": max_f1_stats["f1"],
        "val/max_f1_precision": max_f1_stats["precision"],
        "val/max_f1_recall": max_f1_stats["recall"],
        "val/max_f1_fpr": max_f1_stats["fpr"],
        "val/max_f1_mcc": max_f1_stats["mcc"],
        # Baseline 0.5
        "val/threshold_0_5_f1": baseline_stats["f1"],
        "val/threshold_0_5_precision": baseline_stats["precision"],
        "val/threshold_0_5_recall": baseline_stats["recall"],
        "val/threshold_0_5_fpr": baseline_stats["fpr"],
        # Sweep metric convenience
        "best_f1": max_f1_stats["f1"],
        "model/best_iteration": getattr(clf, "best_iteration_", None),
    }, commit=False)

    # Save model artifact
    best_iter = getattr(clf, "best_iteration_", None)
    num_iter = best_iter if (args.save_best_only and best_iter is not None) else clf.n_estimators
    clf.booster_.save_model("lgbm_model.txt", num_iteration=num_iter)
    art = wandb.Artifact("lightgbm_model", type="model")
    art.add_file("lgbm_model.txt")
    wandb.log_artifact(art)

    wandb.log({"_end": 1}, commit=True)
    wandb.finish()


# --------------------------------------------------------------------------- #
# Baseline stats at an arbitrary fixed threshold (e.g. 0.5)
# --------------------------------------------------------------------------- #
def stats_at_fixed_threshold(y_true, probs, thr):
    y_true = np.asarray(y_true, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    forate = fn / (fn + tn) if (fn + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = 2.0
    f2 = (1 + beta2**2) * precision * recall / (beta2**2 * precision + recall) if (beta2**2 * precision + recall) else 0.0
    beta05 = 0.5
    f05 = (1 + beta05**2) * precision * recall / (beta05**2 * precision + recall) if (beta05**2 * precision + recall) else 0.0
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    balanced_acc = 0.5 * (recall + tnr)
    prevalence = (tp + fn) / total if total else 0.0
    pred_pos_rate = (tp + fp) / total if total else 0.0
    youden_j = recall - fpr
    markedness = precision + npv - 1

    return {
        "label": f"fixed_thr_{thr}",
        "threshold": thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "npv": npv,
        "fdr": fdr,
        "for": forate,
        "f1": f1,
        "f2": f2,
        "f05": f05,
        "mcc": mcc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "prevalence": prevalence,
        "pred_pos_rate": pred_pos_rate,
        "youden_j": youden_j,
        "informedness": youden_j,
        "markedness": markedness,
        "support_pos": tp + fn,
        "support_neg": tn + fp,
        "fallback": False,
    }


if __name__ == "__main__":
    main()


