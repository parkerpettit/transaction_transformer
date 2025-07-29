"""
Logging utilities for pretraining.
"""
import torch
import wandb
from typing import Dict, Any, Optional
from pathlib import Path


class TrainingLogger:
    """Handles W&B logging and progress tracking."""
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], resume: bool = False):
        """
        Initialize W&B logging.
        
        Args:
            project_name: W&B project name
            run_name: W&B run name
            config: Configuration dictionary to log
            resume: Whether to resume existing run
        """
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            resume="allow" if resume else False,
            tags=[
                "pretrain" if config.get("training_type") == "pretrain" else "finetune",
                str(config.get("mode"))
            ],
        )
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """
        Watch model for gradient/parameter logging.
        
        Args:
            model: Model to watch
            log_freq: Frequency of logging
        """
        try:
            wandb.watch(model, log="all", log_freq=log_freq)
        except Exception:
            # Handle torch.compile wrapped models
            pass
    

    
    def log_epoch_metrics(
        self,
        train_loss: float,
        val_loss: float,
        epoch: int,
        epoch_time_min: float,
        learning_rate: float,
        samples_per_epoch: int,
        batches_per_epoch: int,
        best_val_loss: float,
        epochs_without_improvement: int,
        feature_accuracies: Optional[Dict[str, float]] = None
    ):
        """
        Log epoch-level metrics.
        
        Args:
            train_loss: Average training loss for epoch
            val_loss: Validation loss
            epoch: Current epoch number
            epoch_time_min: Time taken for epoch in minutes
            learning_rate: Current learning rate
            samples_per_epoch: Number of samples processed
            batches_per_epoch: Number of batches in epoch
            best_val_loss: Best validation loss so far
            epochs_without_improvement: Consecutive epochs without improvement
            feature_accuracies: Per-feature validation accuracies
        """
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_min": epoch_time_min,
            "learning_rate": learning_rate,
            "samples_per_epoch": samples_per_epoch,
            "batches_per_epoch": batches_per_epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        
        # Add feature-wise accuracies
        if feature_accuracies:
            for feat_name, acc in feature_accuracies.items():
                epoch_log[f"validation_accuracy_{feat_name}"] = acc
            epoch_log["validation_accuracy_overall"] = sum(feature_accuracies.values()) / len(feature_accuracies)
        
       
        wandb.log(epoch_log)
    
    def log_best_model(self, best_val_loss: float):
        """Log new best model metrics."""
        wandb.log({"best_val_loss": best_val_loss})
    
    def log_artifact(self, artifact_path: Path, artifact_name: str, artifact_type: str = "model"):
        """
        Log model artifact to W&B.
        
        Args:
            artifact_path: Path to artifact file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
        """
        if wandb.run is not None:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
    
    def finish(self, exit_code: int = 0):
        """Finish W&B run."""
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
    
    def alert(self, title: str, text: str):
        """Send W&B alert."""
        if self.run is not None:
            self.run.alert(title=title, text=text)
    
