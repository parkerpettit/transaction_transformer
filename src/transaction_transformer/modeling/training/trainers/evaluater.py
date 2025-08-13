"""
Finetuning trainer for transaction transformer.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..base.base_trainer import BaseTrainer
from transaction_transformer.config.config import Config
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.modeling.training.base.checkpoint_manager import (
    CheckpointManager,
)
from transaction_transformer.modeling.training.base.metrics import MetricsTracker
from transaction_transformer.modeling.models import FraudDetectionModel
import wandb
from tqdm import tqdm
from pathlib import Path


class Evaluater:
    """Evaluater for evaluating finetuned models. Evaluates each saved model."""

    def __init__(
        self,
        model_paths: List[str],
        schema: FieldSchema,
        config: Config,
        device: torch.device,
        val_loader: DataLoader,
    ):
        self.model_paths = model_paths
        self.schema = schema
        self.config = config
        self.device = device
        self.val_loader = val_loader
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.schema = schema
        self.config = config
        self.autocast = torch.autocast(device_type=self.device.type, enabled=config.model.training.use_amp, dtype=torch.bfloat16)
        # Initialize components
        self.metrics = MetricsTracker(ignore_index=self.config.model.data.ignore_idx)
        self.metrics.class_names = ["non-fraud", "fraud"]
        self.patience = config.model.training.early_stopping_patience
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.bar_fmt = (
            "{l_bar}{bar:10}| "  # visual bar
            "{n_fmt}/{total_fmt} batches "  # absolute progress
            "({percentage:3.0f}%) | "  # %
            "elapsed: {elapsed} | ETA: {remaining} | "  # timing
            "{rate_fmt} | "  # batches / sec
            "{postfix}"  # losses go here
        )

    @torch.inference_mode()
    def forward_pass(
        self, model: FraudDetectionModel, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch. Returns logits and labels, both of shape (B,)"""
        cat_in = batch["cat"].to(self.device)  #  (B, L, C)
        cont_in = batch["cont"].to(self.device)  # (B, L, F)
        labels = batch["downstream_label"].to(self.device)  # (B,)
        logits = model(
            cat=cat_in,
            cont=cont_in,
        )  # (B,)
        return logits, labels

    @torch.inference_mode()
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for binary classification.
        BCEWithLogitsLoss expects the target tensor to be of floating type with the
        same shape as the input logits. Cast labels to float32 inside the loss
        computation to avoid dtype mismatches (Long vs. Float).
        """
        return self.loss_fn(logits, labels.float())  # (B,)

    @torch.inference_mode()
    def evaluate(self) -> None:
        for i, model_path in enumerate(self.model_paths):
            self.metrics.current_epoch = i + 1
            model = FraudDetectionModel(self.config.model, self.schema)
            self.metrics.wandb_run = wandb.init(
                project=self.config.metrics.wandb_project,
                name=self.config.metrics.run_name,
                config=self.config.to_dict(),
                tags=[self.config.model.training.model_type, f"use_amp={self.config.model.training.use_amp}"],
            )
            if self.metrics.wandb_run:
                artifact = self.metrics.wandb_run.use_artifact(model_path)
                artifact_dir = Path(artifact.download())
                backbone = torch.load(
                    artifact_dir / "backbone.pt", map_location="cpu", weights_only=False
                )
                model.backbone.load_state_dict(backbone["state_dict"], strict=True)
                head = torch.load(
                    artifact_dir / "head.pt", map_location="cpu", weights_only=False
                )
                model.head.load_state_dict(head["state_dict"], strict=True)
            else:
                raise ValueError(
                    "[Evaluater] Warning: no wandb run found, please specify correct model path in evaluater.py"
                )
            model.to(self.device)
            model.eval()

            # Validate for one epoch
            self.metrics.start_epoch()
            self.val_bar = tqdm(
                self.val_loader,
                desc=f"Validation {model_path}",
                bar_format=self.bar_fmt,
                leave=True,
            )
            val_metrics = self.validate_epoch(model)  # type: ignore

            self.metrics.end_epoch(i, "val")
            # Print progress
            print(f"Validation Loss: {val_metrics.get('loss', 0):.4f}")
            print("-" * 50)
            del model

    @torch.inference_mode()
    def validate_epoch(self, model: FraudDetectionModel) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        batch_idx = 0
        for batch in self.val_bar:
            with self.autocast:
                logits, labels = self.forward_pass(model, batch)
                loss = self.compute_loss(logits, labels)  # (B,)
            self.metrics.update_binary_classification(logits, labels)
            total_loss += loss.item()
            self.val_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                }
            )
            if batch_idx % 5 == 0 and self.metrics.wandb_run:
                self.metrics.wandb_run.log(
                    {
                        "val_loss": loss.item(),
                        "epoch": self.metrics.current_epoch,
                    },
                    commit=True,
                )

            batch_idx += 1

        return {"loss": total_loss / len(self.val_loader)}
