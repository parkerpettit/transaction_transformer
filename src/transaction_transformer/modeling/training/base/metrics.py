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


class MetricsTracker:
    """Tracks and manages training metrics with full wandb integration."""
    
    def __init__(self, wandb_run: Optional[Any] = None, class_names: Optional[List[str]] = None):
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
        
        # Loss tracking
        self.total_loss = 0.0
        self.total_samples = 0
    
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
        
        # Reset loss tracking
        self.total_loss = 0.0
        self.total_samples = 0
    
    def update_binary_classification(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        loss: float,
        threshold: float = 0.5
    ) -> None:
        """Update binary classification metrics."""
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        predictions = (probs >= threshold).astype(int)
        
        batch_size = labels.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(labels_np.tolist())
        self.all_predictions.extend(predictions.tolist())
        
        # Update current epoch metrics
        self.current_epoch_metrics["loss"] += loss * batch_size
        self.current_epoch_count += batch_size
    
    def update_transaction_prediction(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch_size: int,
        loss: float
    ) -> None: # note that outputs is a dict mapping feature name to logits. so for example
    # outputs["Merchant_id"] is a tensor of shape (batch_size, vocab_size)
        """Update transaction prediction metrics (per-feature accuracy)."""
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        # Update feature-wise accuracy
        for feature_name, logits_f in outputs.items():
            targets_f = targets[feature_name]
            
            preds_f = logits_f.argmax(dim=1)
            correct = (preds_f == targets_f).sum().item()
            
            self.feature_correct[feature_name] += int(correct)
            self.feature_total[feature_name] += int(targets_f.numel())
        
        # Update current epoch metrics
        self.current_epoch_metrics["loss"] += loss * batch_size
        self.current_epoch_count += batch_size
    
    def _compute_binary_metrics(self) -> Dict[str, float]:
        """Compute comprehensive binary classification metrics."""
        if not self.all_labels:
            return {}
        
        labels_np = np.array(self.all_labels, dtype=np.int8)
        probs_np = np.array(self.all_probs, dtype=np.float32)
        preds_np = np.array(self.all_predictions, dtype=np.int8)
        
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
        
        # AUC metrics
        try:
            roc_auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            roc_auc = float("nan")
        
        try:
            pr_auc = average_precision_score(labels_np, probs_np)
        except ValueError:
            pr_auc = float("nan")
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "class_acc_0": class_acc[0],
            "class_acc_1": class_acc[1],
            "avg_loss": self.total_loss / self.total_samples if self.total_samples > 0 else 0.0
        }
    
    def _compute_transaction_metrics(self) -> Dict[str, float]:
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
            metrics["avg_accuracy"] = avg_acc
        
        metrics["avg_loss"] = self.total_loss / self.total_samples if self.total_samples > 0 else 0.0
        
        return metrics
    
    def _log_binary_plots(self, labels_np: np.ndarray, probs_np: np.ndarray, preds_np: np.ndarray) -> None:
        """Log binary classification plots to wandb."""
        if not self.wandb_run:
            return
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix")
        self.wandb_run.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        
        # ROC and PR curves
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
        
        self.wandb_run.log({
            "roc_curve": roc_plot,
            "pr_curve": pr_plot
        })
    
    def _log_transaction_plots(self) -> None:
        """Log transaction prediction plots to wandb."""
        if not self.wandb_run or not self.feature_total:
            return
        
        # Feature accuracy bar chart
        feature_names = list(self.feature_total.keys())
        accuracies = [self.feature_correct[name] / self.feature_total[name] 
                     for name in feature_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(feature_names, accuracies)
        ax.set_title("Per-Feature Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.wandb_run.log({"feature_accuracy": wandb.Image(fig)})
        plt.close(fig)
    
    def end_epoch(self, epoch: int, mode: str = "train") -> Dict[str, float]:
        """End epoch and return averaged metrics."""
        if self.total_samples == 0:
            return {}
        
        # Compute metrics based on what we've been tracking
        if self.all_labels:  # Binary classification
            metrics = self._compute_binary_metrics()
            labels_np = np.array(self.all_labels, dtype=np.int8)
            probs_np = np.array(self.all_probs, dtype=np.float32)
            preds_np = np.array(self.all_predictions, dtype=np.int8)
            self._log_binary_plots(labels_np, probs_np, preds_np)
        else:  # Transaction prediction
            metrics = self._compute_transaction_metrics()
            self._log_transaction_plots()
        
        # Add epoch and mode info
        metrics["epoch"] = float(epoch)
        # Note: mode is a string, not a float, but we need it for internal tracking
        # We'll exclude it from the return dict to maintain float-only values
        
        # Log to wandb
        if self.wandb_run:
            wandb_metrics = {f"{mode}_{k}": v for k, v in metrics.items() 
                           if k not in ["epoch", "mode"]}
            self.wandb_run.log(wandb_metrics)
        
        # Store in history
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to wandb."""
        if self.wandb_run:
            self.wandb_run.log(metrics)
    
    def get_best_metric(self, metric_name: str, maximize: bool = True) -> Optional[float]:
        """Get the best value for a given metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        if maximize:
            return max(values)
        else:
            return min(values)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get the history of a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.current_epoch_metrics.clear()
        self.current_epoch_count = 0
        self.all_probs.clear()
        self.all_labels.clear()
        self.all_predictions.clear()
        self.feature_correct.clear()
        self.feature_total.clear()
        self.total_loss = 0.0
        self.total_samples = 0 