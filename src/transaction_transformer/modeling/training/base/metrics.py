"""
Metrics tracking for transaction transformer with full wandb integration.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)
import wandb
from transaction_transformer.data.preprocessing.schema import FieldSchema


class MetricsTracker:
    """Tracks and manages training metrics with full wandb integration."""

    def __init__(
        self,
        wandb_run: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100,
    ):
        """
        Initialize metrics tracker.

        Args:
            wandb_run: wandb run object for logging
            class_names: List of class names for binary classification (e.g., ["non-fraud", "fraud"])
        """
        self.wandb_run = wandb_run
        self.class_names = class_names or ["non-fraud", "fraud"]
        self.logger = logging.getLogger(__name__)
        # Training metrics
        self.metrics = defaultdict(list)
        self.current_epoch = 0
        self.start_time = None

        # Binary classification tracking
        self.all_probs: List[float] = []
        self.all_labels: List[int] = []
        self.all_predictions: List[int] = []

        # Per-feature tracking (for transaction prediction)
        self.feature_correct: Dict[str, int] = defaultdict(int)
        self.feature_total: Dict[str, int] = defaultdict(int)
        self.feature_names: List[str] = []

        self.ignore_index = ignore_index

    def start_epoch(self) -> None:
        """Start tracking metrics for a new epoch."""
        self.start_time = time.time()

        # Reset binary classification tracking
        self.all_probs.clear()
        self.all_labels.clear()
        self.all_predictions.clear()

        # Reset per-feature tracking
        self.feature_correct.clear()
        self.feature_total.clear()

        self.metrics.clear()

    def update_binary_classification(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update binary classification metrics."""
        # Ensure stable dtype for NumPy under AMP (bf16/fp16) by casting to float32
        probs = torch.sigmoid(logits).to(dtype=torch.float32).cpu().numpy()
        labels_np = labels.to(dtype=torch.int64).cpu().numpy()

        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(labels_np.tolist())

    def update_transaction_prediction(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> (
        None
    ):  # note that outputs is a dict mapping feature name to logits. so for example
        # outputs["Merchant_id"] is a tensor of shape (batch_size, L, merchant_id_vocab_size) for both MLM and AR
        """Update transaction prediction metrics (per-feature accuracy)."""

        # Update feature-wise accuracy
        for feature_name, logits_f in logits.items():
            targets_f = targets[feature_name]

            # Both AR and MLM now return (B, L, V) logits
            # AR: compute accuracy on shifted positions (non-ignore)
            # MLM: compute accuracy on masked positions (non-ignore)
            preds_f = logits_f.argmax(dim=2)  # (B, L)

            # Only calculate accuracy on non-ignore positions
            mask = targets_f != self.ignore_index
            if mask.any():
                correct = (preds_f[mask] == targets_f[mask]).sum().item()
                total_positions = mask.sum().item()
            else:
                correct = 0
                total_positions = 0

            self.feature_correct[feature_name] += int(correct)
            self.feature_total[feature_name] += int(total_positions)

    def print_sample_predictions(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        schema: FieldSchema,
    ) -> None:
        """
        Pretty print predictions for a single sample (all features). Logits is a dict of feature name to logits tensor of shape (B, L, V_field) for both MLM and AR.
        Targets is a dict of feature name to targets tensor of shape (B, L) for both MLM and AR.
        Schema is the FieldSchema object.
        """

        # Helper to decode categorical codes
        def decode_cat(code, feat_name):
            return schema.cat_encoders[feat_name].inv[code]

        # Get batch size from any feature
        batch_size = targets[list(targets.keys())[0]].size(0)

        if batch_size == 0:
            self.logger.info("No samples to print.")
            return

        # Both AR and MLM now have (B, L, V) format
        # Scan through each batch, then each row
        for batch_idx in range(batch_size):
            L = logits[list(logits.keys())[0]].shape[1]
            for row_idx in range(L):
                # Check if any features are valid targets in this row (non-ignore)
                valid_feats = []
                for feat in logits.keys():
                    if targets[feat][batch_idx, row_idx] != self.ignore_index:
                        valid_feats.append(feat)

                if valid_feats:
                    self.logger.info(f"\nBatch {batch_idx}, Row {row_idx} (valid targets):")
                    self.logger.info(
                        f"{'Feature':<15} {'Target':<25} {'Predicted':<25} {'Correct'}"
                    )
                    self.logger.info("-" * 70)
                    # Print predictions for all valid features in this row
                    for feat in valid_feats:
                        pred_code = logits[feat][batch_idx, row_idx].argmax().item()
                        tgt_code = targets[feat][batch_idx, row_idx].item()
                        correct = "YES" if pred_code == tgt_code else "NO"

                        if feat in schema.cat_features:
                            tgt_str = decode_cat(tgt_code, feat)
                            pred_str = decode_cat(pred_code, feat)
                            self.logger.info(f"{feat:<15} {tgt_str:<25} {pred_str:<25} {correct}")
                        else:
                            self.logger.info(
                                f"{feat:<15} {tgt_code:<25} {pred_code:<25} {correct}"
                            )
                    self.logger.info("-" * 40)
                    return  # Break after finding first valid row

    def threshold_search(
        self, target_fpr: float, probs: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute a threshold achieving the target FPR using the empirical negative-score quantile.

        The target FPR is given as a percentage (e.g., 0.1 for 0.1%). We choose a threshold
        just above the cutpoint in the sorted negative scores so that, under the rule
        (pred = probs >= threshold), at most k_allowed negatives are predicted positive.

        Args:
            target_fpr: float, target FPR in percent (e.g., 0.1 means 0.1%).
            probs: np.ndarray, predicted probabilities of shape (N,).
            labels: np.ndarray, ground-truth labels of shape (N,), where 0 denotes negatives.

        Returns:
            float: threshold in [0, 1]. If no negatives exist, returns 1.0.
        """
        # Convert percent to fraction
        target = float(target_fpr) / 100.0

        # Extract negative scores
        neg = probs[labels == 0]
        if neg.size == 0:
            return 1.0

        # Sort negatives ascending once
        neg_sorted = np.sort(neg.astype(np.float64))
        N = int(neg_sorted.size)

        # Max allowed false positives among negatives
        k_allowed = int(np.floor(target * N + 1e-12))

        if k_allowed <= 0:
            # No negatives allowed positive: set threshold strictly above max negative
            base = neg_sorted[-1]
            return float(np.nextafter(base, np.float64(np.inf)))

        if k_allowed >= N:
            # All negatives allowed positive: practically threshold at 0.0
            return 0.0

        # Keep exactly N - k_allowed negatives strictly below threshold
        idx = N - k_allowed - 1
        base = neg_sorted[idx]
        # Step to the next representable float above the base to ensure <= k_allowed FPs
        thr = np.nextafter(base, np.float64(np.inf))
        return float(thr)

    def _compute_binary_metrics(self) -> Dict[str, float]:
        """Compute comprehensive binary classification metrics at multiple thresholds."""
        if not self.all_labels:
            return {}

        # Compute predictions
        labels_np = np.array(self.all_labels, dtype=np.int8)
        probs_np = np.array(self.all_probs, dtype=np.float32)

        # Define thresholds for target FPRs
        target_fprs = [
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
        ]  # 0.1%, 0.05%, 0.01%, 0.005%, 0.001% FPR
        thresholds = [
            self.threshold_search(target_fpr, probs_np, labels_np)
            for target_fpr in target_fprs
        ]

        # Store metrics for each threshold
        metrics = {}

        # rows_for_table = []
        for target_fpr, threshold in zip(target_fprs, thresholds):
            preds_np = (probs_np >= threshold).astype(np.int8)

            # Basic metrics
            accuracy = (preds_np == labels_np).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds_np, average="binary", zero_division=0.0 # type: ignore
            )

            # Class-wise accuracy
            class_acc = {
                0: (
                    float((preds_np[labels_np == 0] == 0).mean())
                    if (labels_np == 0).any()
                    else 0.0
                ),
                1: (
                    float((preds_np[labels_np == 1] == 1).mean())
                    if (labels_np == 1).any()
                    else 0.0
                ),
            }

            # Store metrics with threshold-specific keys
            suffix = f"@fpr{target_fpr:.3f}%"
            metrics[f"threshold{suffix}"] = float(threshold)
            metrics[f"overall_accuracy{suffix}"] = float(accuracy)
            metrics[f"precision{suffix}"] = float(precision)
            metrics[f"recall{suffix}"] = float(recall)
            metrics[f"f1{suffix}"] = float(f1)
            metrics[f"non-fraud_acc{suffix}"] = class_acc[0]
            metrics[f"fraud_acc{suffix}"] = class_acc[1]

            self._log_confusion_matrix(labels_np, preds_np, target_fpr)

            # rows_for_table.append([
            #     int(self.current_epoch),
            #     float(target_fpr),
            #     float(threshold),
            #     float(precision),
            #     float(recall),
            #     float(f1),
            #     float(accuracy),
            #     float(class_acc[0]),
            #     float(class_acc[1]),
            # ])

        # AUC metrics (not threshold-dependent)
        try:
            roc_auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            roc_auc = float("nan")

        try:
            pr_auc = average_precision_score(labels_np, probs_np)
        except ValueError:
            pr_auc = float("nan")

        metrics["roc_auc"] = float(roc_auc)
        metrics["pr_auc"] = float(pr_auc)
        self.metrics.update(metrics)
        self._log_roc_pr_curves(labels_np, probs_np)

        # # Log raw prediction probabilities and labels for post-hoc thresholding
        # if self.wandb_run:
        #     pred_rows = [[int(self.current_epoch), int(i), float(p), int(l)] for i, (p, l) in enumerate(zip(probs_np, labels_np))]
        #     pred_tbl = wandb.Table(columns=["epoch", "index", "prob", "label"], data=pred_rows)
        #     self.wandb_run.log({
        #         "epoch": self.current_epoch,
        #         "val_predictions_table": pred_tbl,
        #     })

        if self.wandb_run:
            # Precision/Recall/F1 vs threshold curve (dense sampling)
            thresholds_dense = np.linspace(0.0, 1.0, 201)
            prf_rows = []
            prec_list, rec_list, f1_list = [], [], []
            for th in thresholds_dense:
                preds = (probs_np >= th).astype(np.int8)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_np, preds, average="binary", zero_division=0.0 # type: ignore
                )
                prf_rows.append(
                    [
                        int(self.current_epoch),
                        float(th),
                        float(precision),
                        float(recall),
                        float(f1),
                    ]
                )
                prec_list.append(float(precision))
                rec_list.append(float(recall))
                f1_list.append(float(f1))
            prf_tbl = wandb.Table(
                columns=["epoch", "threshold", "precision", "recall", "f1"],
                data=prf_rows,
            )
            self.wandb_run.log(
                {
                    "epoch": self.current_epoch,
                    "val_prf_curve_table": prf_tbl,
                    "val_prf_curve": wandb.plot.line_series(
                        xs=thresholds_dense.tolist(),
                        ys=[prec_list, rec_list, f1_list],
                        keys=["precision", "recall", "f1"],
                        title="Validation Precision/Recall/F1 vs Threshold",
                        xname="threshold",
                    ),
                }
            )

        # # Log a compact threshold metrics table per epoch
        # if self.wandb_run and rows_for_table:
        #     tbl = wandb.Table(columns=[
        #         "epoch", "target_fpr_percent", "threshold", "precision", "recall", "f1",
        #         "overall_accuracy", "non_fraud_acc", "fraud_acc",
        #     ], data=rows_for_table)
        #     self.wandb_run.log({
        #         "epoch": self.current_epoch,
        #         "val_threshold_metrics_table": tbl,
        #     })
        return metrics

    def _compute_transaction_metrics(self) -> None:
        """Compute transaction prediction metrics."""
        metrics = {}

        # Per-feature accuracy
        for feature_name in self.feature_total.keys():
            if self.feature_total[feature_name] > 0:
                acc = (
                    self.feature_correct[feature_name]
                    / self.feature_total[feature_name]
                )
                metrics[f"acc_{feature_name}"] = acc

        # Average accuracy across features
        if self.feature_total:
            total_correct = sum(self.feature_correct.values())
            total_positions = sum(self.feature_total.values())
            if total_positions > 0:
                avg_acc = total_correct / total_positions
                metrics["avg_acc"] = avg_acc

        self.metrics.update(metrics)
        # Log a per-feature accuracy table
        # if self.wandb_run and self.feature_total:
        #     rows = []
        #     for feature_name in self.feature_total.keys():
        #         total_positions = self.feature_total[feature_name]
        #         acc = self.feature_correct[feature_name] / total_positions if total_positions > 0 else 0.0
        #         rows.append([int(self.current_epoch), feature_name, float(acc), int(total_positions)])
        #     tbl = wandb.Table(columns=["epoch", "feature", "accuracy", "valid_positions"], data=rows)
        #     self.wandb_run.log({
        #         "epoch": self.current_epoch,
        #         "transaction_feature_accuracy_table": tbl,
        #     })

    def _log_confusion_matrix(
        self, labels_np: np.ndarray, preds_np: np.ndarray, target_fpr: float
    ) -> None:
        """Log confusion matrix to wandb for a specific threshold (target FPR)."""
        if not self.wandb_run:
            return

        cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average="binary", zero_division=0.0 # type: ignore
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix @ FPR {target_fpr:.3f}%")

        # Add precision, recall, and f1 underneath the confusion matrix
        metrics_text = (
            f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
        )
        # Place the metrics text below the confusion matrix
        fig.text(0.45, 0.0, metrics_text, ha="center", va="bottom", fontsize=10)

        # Create unique key for each confusion matrix based on target FPR with % symbol
        cm_key = f"confusion_matrix_fpr_{target_fpr:.3f}%".replace(".", "_").replace(
            "%", "pct"
        )
        self.wandb_run.log(
            {
                "epoch": self.current_epoch,
                cm_key: wandb.Image(fig),
            }
        )
        plt.close(fig)

    def _log_roc_pr_curves(self, labels_np: np.ndarray, probs_np: np.ndarray) -> None:
        """Log ROC and PR curves to wandb (not threshold-dependent)."""
        if not self.wandb_run:
            return

        # Plot ROC and PR curves using sklearn and matplotlib, then log to wandb as images

        probas_2d = np.vstack([1 - probs_np, probs_np]).T

        roc_plot = wandb.plot.roc_curve(
            y_true=labels_np.tolist(),
            y_probas=probas_2d.tolist(),
            labels=self.class_names,
        )
        pr_plot = wandb.plot.pr_curve(
            y_true=labels_np.tolist(),
            y_probas=probas_2d.tolist(),
            labels=self.class_names,
        )

        # Also log a confusion matrix at threshold 0.5 (FPR not computed, just threshold)
        preds_05 = (probs_np >= 0.5).astype(int)
        cm_05 = confusion_matrix(labels_np, preds_05, labels=[0, 1])
        fig_05, ax_05 = plt.subplots(figsize=(8, 6))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_05, average="binary", zero_division=0.0 # type: ignore
        )

        disp_05 = ConfusionMatrixDisplay(cm_05, display_labels=self.class_names)
        disp_05.plot(ax=ax_05, cmap="Blues", values_format="d")
        ax_05.set_title("Confusion Matrix @ threshold 0.5")
        cm_key_05 = "confusion_matrix_thresh_0_5"
        metrics_text = f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"  # Place the metrics text below the confusion matrix
        fig_05.text(0.45, 0.0, metrics_text, ha="center", va="bottom", fontsize=10)
        self.wandb_run.log(
            {
                "epoch": self.current_epoch,
                cm_key_05: wandb.Image(fig_05),
                "roc_curve": roc_plot,
                "pr_curve": pr_plot,
                "val_precision@0.5_threshold": float(precision),
                "val_recall@0.5_threshold": float(recall),
                "val_f1@0.5_threshold": float(f1),
            }
        )
        plt.close("all")

    def _log_transaction_plots(self, model_type: str) -> None:
        """Log transaction prediction plots to wandb."""
        if not self.wandb_run or not self.feature_total:
            return

        # Feature accuracy bar chart
        feature_names = list(self.feature_total.keys())
        accuracies = []
        for name in feature_names:
            if self.feature_total[name] > 0:
                acc = self.feature_correct[name] / self.feature_total[name]
                accuracies.append(acc)
            else:
                accuracies.append(0.0)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(feature_names, accuracies)
        ax.set_title(f"{model_type.capitalize()} Per-Feature Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        self.wandb_run.log(
            {
                "epoch": self.current_epoch,
                f"{model_type}_feature_accuracy": wandb.Image(fig),
            }
        )
        plt.close(fig)

    def end_epoch(self, epoch: int, model_type: str = "train") -> None:
        """End epoch and return averaged metrics."""

        # Compute heavy metrics during validation only (avoid noise/overhead during training)
        if model_type == "val":
            if self.all_labels:  # Binary classification (finetune)
                self._compute_binary_metrics()
            else:  # Transaction prediction (pretraining)
                self._compute_transaction_metrics()
                self._log_transaction_plots(model_type)

        # Add epoch and model_type info
        # Log to wandb
        if self.wandb_run:
            wandb_metrics: Dict[str, Any] = {
                f"{model_type}_{k}": v for k, v in self.metrics.items()
            }
            wandb_metrics["epoch"] = int(epoch)
            self.logger.info(f"Logging metrics to wandb for {model_type} epoch {epoch}")
            self.wandb_run.log(wandb_metrics)
            self.logger.info(f"Metrics logged to wandb for {model_type} epoch {epoch}")

        # Store in history
        return
