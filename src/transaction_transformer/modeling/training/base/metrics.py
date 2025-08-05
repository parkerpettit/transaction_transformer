"""
Metrics tracking for transaction transformer with full wandb integration.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
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
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema

class MetricsTracker:
    """Tracks and manages training metrics with full wandb integration."""
    
    def __init__(self, wandb_run: Optional[Any] = None, class_names: Optional[List[str]] = None, ignore_index: int = -100):
        """
        Initialize metrics tracker.
        
        Args:
            wandb_run: wandb run object for logging
            class_names: List of class names for binary classification (e.g., ["non-fraud", "fraud"])
        """
        self.wandb_run = wandb_run
        self.class_names = class_names or ["non-fraud", "fraud"]
        
        # Training metrics
        self.metrics = defaultdict(list)
        self.current_epoch_metrics = defaultdict(float)
        self.current_epoch_count = 0
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
        self.current_epoch_metrics.clear()
        self.current_epoch_count = 0
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
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(labels_np.tolist())

    
    def update_transaction_prediction(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None: # note that outputs is a dict mapping feature name to logits. so for example
    # outputs["Merchant_id"] is a tensor of shape (batch_size, L, merchant_id_vocab_size) if mlm, (batch_size, merchant_id_vocab_size) if ar
        """Update transaction prediction metrics (per-feature accuracy)."""
        
        # Update feature-wise accuracy
        for feature_name, logits_f in logits.items():
            targets_f = targets[feature_name]
            
            # Determine the correct dimension for argmax based on tensor shapes
            if logits_f.dim() == 2:  # AR: (B, V)
                preds_f = logits_f.argmax(dim=1)
                # For AR, all positions are valid targets
                correct = (preds_f == targets_f).sum().item()
                total_positions = targets_f.numel()
            elif logits_f.dim() == 3:  # MLM: (B, L, V)
                preds_f = logits_f.argmax(dim=2)
                # For MLM, only calculate accuracy on masked positions (non-ignore)
                mask = targets_f != self.ignore_index
                if mask.any():
                    correct = (preds_f[mask] == targets_f[mask]).sum().item()
                    total_positions = mask.sum().item()
                    # print(
                    #     f"[{feature_name}] MLM: masked positions = {mask.sum().item()} / total = {targets_f.numel()} "
                    #     f"({mask.sum().item()/targets_f.numel():.2%} masked); "
                    #     f"ignore_index count = {(targets_f == self.ignore_index).sum().item()}; "
                    #     f"accuracy: {correct / total_positions:.2%}"
                    # )
                else:
                    correct = 0
                    total_positions = 0
            else:
                raise ValueError(f"Unexpected logits shape for {feature_name}: {logits_f.shape}")
            
            self.feature_correct[feature_name] += int(correct)
            self.feature_total[feature_name] += int(total_positions)

    def print_sample_predictions(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        schema: FieldSchema,
    ) -> None:
        """
        Pretty print predictions for a single sample (all features). Logits is a dict of feature name to logits tensor of shape (B, L, V_field) if mlm, (B, V_field) if ar.
        Targets is a dict of feature name to targets tensor of shape (B, L) if mlm, (B,) if ar.
        Schema is the FieldSchema object.
        """
        # Helper to decode categorical codes
        def decode_cat(code, feat_name):
            return schema.cat_encoders[feat_name].inv[code]

        # Get batch size from any feature
        batch_size = targets[list(targets.keys())[0]].size(0)
        
        if batch_size == 0:
            print("No samples to print.")
            return

        # Determine if this is MLM model_type (any feature logits dim == 3)
        is_mlm = any(logits[feat].dim() == 3 for feat in logits.keys())

        if is_mlm:
            # Scan through each batch, then each row
            for batch_idx in range(batch_size):
                L = logits[list(logits.keys())[0]].shape[1]
                for row_idx in range(L):
                    # Check if any features are masked in this row
                    masked_feats = []
                    for feat in logits.keys():
                        if targets[feat][batch_idx, row_idx] != self.ignore_index:
                            masked_feats.append(feat)
                    
                    if masked_feats:
                        print(f"\nBatch {batch_idx}, Row {row_idx} (masked positions):")
                        print(f"{'Feature':<15} {'Target':<25} {'Predicted':<25} {'Correct'}")
                        print("─" * 70)
                        # Print predictions for all masked features in this row
                        for feat in masked_feats:
                            pred_code = logits[feat][batch_idx, row_idx].argmax().item()
                            tgt_code = targets[feat][batch_idx, row_idx].item()
                            correct = "YES" if pred_code == tgt_code else "NO"
                            
                            if feat in schema.cat_features:
                                tgt_str = decode_cat(tgt_code, feat)
                                pred_str = decode_cat(pred_code, feat)
                                print(f"{feat:<15} {tgt_str:<25} {pred_str:<25} {correct}")
                            else:
                                print(f"{feat:<15} {tgt_code:<25} {pred_code:<25} {correct}")
                        print("─" * 40)
                        return  # Break after finding first masked row
        else:
            # AR model_type: print all features for the first sample
            sample_idx = 0
            print(f"\nSample {sample_idx} (AR model_type):")
            print(f"{'Feature':<15} {'Target':<25} {'Predicted':<25} {'Correct'}")
            print("─" * 70)
            for feat in logits.keys():
                pred_code = logits[feat][sample_idx].argmax().item()
                tgt_code = targets[feat][sample_idx].item()
                correct = "YES" if pred_code == tgt_code else "NO"
                
                if feat in schema.cat_features:
                    tgt_str = decode_cat(tgt_code, feat)
                    pred_str = decode_cat(pred_code, feat)
                    print(f"{feat:<15} {tgt_str:<25} {pred_str:<25} {correct}")
                else:
                    print(f"{feat:<15} {tgt_code:<25} {pred_code:<25} {correct}")
            print("─" * 40)


    def threshold_search(self, target_fpr: float, probs: np.ndarray, labels: np.ndarray) -> float:
        """ Binary search for the threshold results in a target FPR. Target FPR should be given as
            a percentage, e.g. 0.01 for 0.01% FPR.
        Args:
            target_fpr: float, target FPR
            probs: np.ndarray, probabilities of shape (B,)
            labels: np.ndarray, labels of shape (B,)
        Returns:
            float, best threshold
        """
    
        best_threshold = 1.0
        low = 0.0
        high = 1.0
        target_fpr = target_fpr / 100
        
        while low <= high:
            mid = (low + high) / 2
            preds = (probs >= mid).astype(int)
            
            # Calculate true FPR
            neg_samples = labels == 0
            if neg_samples.sum() > 0:
                fpr = ((preds == 1) & neg_samples).sum() / neg_samples.sum()
            else:
                fpr = 0.0
                
            if fpr <= target_fpr:
                best_threshold = mid
                high = mid - 1e-6
            else:
                low = mid + 1e-6
    
        return best_threshold

    def _compute_binary_metrics(self) -> Dict[str, float]:
        """Compute comprehensive binary classification metrics at multiple thresholds."""
        if not self.all_labels:
            return {}

        # Compute predictions
        labels_np = np.array(self.all_labels, dtype=np.int8)
        probs_np = np.array(self.all_probs, dtype=np.float32)

        # Define thresholds for target FPRs
        target_fprs = [0.1, 0.05, 0.01, 0.005, 0.001]  # 0.1%, 0.05%, 0.01%, 0.005%, 0.001% FPR
        thresholds = [self.threshold_search(target_fpr, probs_np, labels_np) for target_fpr in target_fprs]

        # Store metrics for each threshold
        metrics = {}

        for target_fpr, threshold in zip(target_fprs, thresholds):
            preds_np = (probs_np >= threshold).astype(np.int8)

            # Basic metrics
            accuracy = (preds_np == labels_np).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds_np, average="binary", zero_division="warn"
            )

            # Class-wise accuracy
            class_acc = {
                0: float((preds_np[labels_np == 0] == 0).mean()) if (labels_np == 0).any() else 0.0,
                1: float((preds_np[labels_np == 1] == 1).mean()) if (labels_np == 1).any() else 0.0,
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
        return metrics
    
    def _compute_transaction_metrics(self) -> None:
        """Compute transaction prediction metrics."""
        metrics = {}
        
        # Per-feature accuracy
        for feature_name in self.feature_total.keys():
            if self.feature_total[feature_name] > 0:
                acc = self.feature_correct[feature_name] / self.feature_total[feature_name]
                metrics[f"acc_{feature_name}"] = acc
        
        # Average accuracy across features
        if self.feature_total:
            avg_acc = sum(self.feature_correct.values()) / sum(self.feature_total.values())
            metrics["avg_acc"] = avg_acc
        
        self.metrics.update(metrics)
    
    def _log_confusion_matrix(self, labels_np: np.ndarray, preds_np: np.ndarray, target_fpr: float) -> None:
        """Log confusion matrix to wandb for a specific threshold (target FPR)."""
        if not self.wandb_run:
            return

        cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix @ FPR {target_fpr:.3f}%")
        
        # Create unique key for each confusion matrix based on target FPR with % symbol
        cm_key = f"confusion_matrix_fpr_{target_fpr:.3f}%".replace(".", "_").replace("%", "pct")
        self.wandb_run.log({cm_key: wandb.Image(fig)})
        plt.close(fig)

    def _log_roc_pr_curves(self, labels_np: np.ndarray, probs_np: np.ndarray) -> None:
        """Log ROC and PR curves to wandb (not threshold-dependent)."""
        if not self.wandb_run:
            return

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
        disp_05 = ConfusionMatrixDisplay(cm_05, display_labels=self.class_names)
        disp_05.plot(ax=ax_05, cmap="Blues", values_format="d")
        ax_05.set_title("Confusion Matrix @ threshold 0.5")
        cm_key_05 = "confusion_matrix_thresh_0_5"
        self.wandb_run.log({cm_key_05: wandb.Image(fig_05)})
        plt.close(fig_05)
        self.wandb_run.log({
            "roc_curve": roc_plot,
            "pr_curve": pr_plot
        })
    
    def _log_transaction_plots(self, model_type: str) -> None:
        """Log transaction prediction plots to wandb."""
        if not self.wandb_run or not self.feature_total:
            return
        
        # Feature accuracy bar chart
        feature_names = list(self.feature_total.keys())
        accuracies = [self.feature_correct[name] / self.feature_total[name] 
                     for name in feature_names]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(feature_names, accuracies)
        ax.set_title(f"{model_type.capitalize()} Per-Feature Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.wandb_run.log({f"{model_type}_feature_accuracy": wandb.Image(fig)})
        plt.close(fig)
    
    def end_epoch(self, epoch: int, model_type: str = "train") -> None:
        """End epoch and return averaged metrics."""
        
        # Compute metrics based on what we've been tracking
        if self.all_labels:  # Binary classification
            print(f"Computing and logging binary metrics for {model_type} epoch {epoch}")
            self._compute_binary_metrics()
            print(f"Binary metrics computed and logged for {model_type} epoch {epoch}")
        else:  # Transaction prediction
            print(f"Computing transaction metrics for {model_type} epoch {epoch}")
            self._compute_transaction_metrics()
            print(f"Transaction metrics computed for {model_type} epoch {epoch}")
            print(f"Logging transaction plots for {model_type} epoch {epoch}")
            self._log_transaction_plots(model_type)
            print(f"Transaction plots logged for {model_type} epoch {epoch}")
        
        # Add epoch and model_type info
        
        # Note: model_type is a string, not a float, but we need it for internal tracking
        # We'll exclude it from the return dict to maintain float-only values
        # Log to wandb
        if self.wandb_run:
            wandb_metrics = {f"{model_type}_{k}": v for k, v in self.metrics.items()}
            print(f"Logging metrics to wandb for {model_type} epoch {epoch}")
            self.wandb_run.log(wandb_metrics)
            print(f"Metrics logged to wandb for {model_type} epoch {epoch}")
        
        # Store in history
        return
    
 
    
