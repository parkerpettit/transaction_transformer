"""
Core training logic for pretraining.
"""
import torch
import torch.nn as nn
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm
import torch.nn.functional as F

from configs.config import ModelConfig
from models.transformer.transformer_model import TransactionModel
from data.dataset import slice_batch
from evaluation.evaluate import evaluate
from utils.utils import save_ckpt
from configs.paths import ProjectPaths
from utils.masking import (
    create_field_and_row_mask,
    apply_field_level_categorical_masking, 
    apply_field_level_continuous_masking,
    compute_field_level_mlm_loss
)
from training.logging_utils import TrainingLogger


class PretrainTrainer:
    """Core trainer for pretraining TransactionModel."""
    
    def __init__(
        self,
        model: TransactionModel,
        config: ModelConfig,
        args: Any,
        device: torch.device,
        cat_features: List[str],
        cont_features: List[str],
        logger: TrainingLogger,
        encoders: Optional[Dict[str, Dict[str, Any]]] = None,
        qparams: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Model configuration
            args: Training arguments
            device: Device to train on
            cat_features: Categorical feature names
            cont_features: Continuous feature names
            logger: Training logger
        """
        self.model = model
        self.config = config
        self.args = args
        self.device = device
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.logger = logger
        self.encoders = encoders
        self.qparams = qparams
        
        # Setup training components
        self.vocab_sizes = list(config.cat_vocab_sizes.values())
        self.criterion_cat = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_cont = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # Performance optimizations
        self.scaler = GradScaler() if args.use_mixed_precision else None
        
        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.patience = 2
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Progress bar
        bar_fmt = (
            "{l_bar}{bar:25}| "
            "{n_fmt}/{total_fmt} batches "
            "({percentage:3.0f}%) | "
            "elapsed: {elapsed} | ETA: {remaining} | "
            "{rate_fmt} | "
            "{postfix}"
        )
        
        prog_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.args.total_epochs}",
            unit="batch",
            total=len(train_loader),
            bar_format=bar_fmt,
            ncols=200,
            leave=True,
        )
        
        for batch_idx, batch in enumerate(prog_bar):
            if batch_idx > 50:
                break
            loss, batch_size, extra_metrics = self._train_batch(batch)
            
            total_loss = total_loss + loss * batch_size
            total_samples = total_samples + batch_size
                
            prog_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "mode": self.args.mode,
            })
        
        return total_loss / total_samples
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Train on a single batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, batch_size, extra_metrics)
        """
        # Prepare batch data
        mask = None
        if self.args.mode in ["masked", "mlm"]:
            batch_items = self._prepare_mlm_batch(batch)
            if len(batch_items) == 6:
                cat_input, cont_inp, cat_tgt, cont_tgt, qtarget, mask = batch_items  # type: ignore
            else:
                cat_input, cont_inp, cat_tgt, cont_tgt, qtarget = batch_items  # type: ignore
        else:
            cat_input, cont_inp, cat_tgt, cont_tgt, qtarget = self._prepare_ar_batch(batch)
        
        batch_size = cat_input.size(0)
        self.optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if self.args.use_mixed_precision and self.scaler is not None:
            with autocast('cuda'):
                if self.args.mode in ["masked", "mlm"]:
                    loss, extra_metrics = self._forward_mlm(
                        cat_input, cont_inp, cat_tgt, cont_tgt, qtarget, mask
                    )
                else:
                    loss, extra_metrics = self._forward_ar(
                        cat_input, cont_inp, cat_tgt, cont_tgt, qtarget
                    )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.args.mode in ["masked", "mlm"]:
                loss, extra_metrics = self._forward_mlm(
                    cat_input, cont_inp, cat_tgt, cont_tgt, qtarget, mask
                )
            else:
                loss, extra_metrics = self._forward_ar(
                    cat_input, cont_inp, cat_tgt, cont_tgt, qtarget
                )
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), batch_size, extra_metrics
    
    def _prepare_mlm_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Prepare batch for MLM training with efficient collator."""
        # With the new collator, the batch already contains masked inputs and labels
        if "cat_input" in batch:
            # New efficient collator format
            cat_input = batch["cat_input"].to(self.device)
            cont_inp = batch["cont_input"].to(self.device)
            cat_labels = batch["cat_labels"].to(self.device)
            cont_labels = batch["cont_labels"].to(self.device)
            mask = batch["mask"].to(self.device)
            qtarget_labels = batch.get("qtarget_labels")
            if qtarget_labels is not None:
                qtarget_labels = qtarget_labels.to(self.device)
            
            return cat_input, cont_inp, cat_labels, cont_labels, qtarget_labels, mask
        else:
            # Fallback to old format for compatibility
            cat_input = batch["cat"].to(self.device)
            cont_inp = batch["cont"].to(self.device)
            cat_tgt = batch["cat"].to(self.device)  # Full sequence as targets
            cont_tgt = batch["cont"].to(self.device)
            qtarget = batch.get("qtarget")
            if qtarget is not None:
                qtarget = qtarget.to(self.device)
            
            return cat_input, cont_inp, cat_tgt, cont_tgt, qtarget, None
    
    def _prepare_ar_batch(self, batch: Dict[str, torch.Tensor]):
        """Prepare batch for AR training."""
        batch_items = slice_batch(batch)
        if len(batch_items) == 6:
            cat_input, cont_inp, cat_tgt, cont_tgt, qtarget, _ = (t.to(self.device) for t in batch_items)
        else:
            cat_input, cont_inp, cat_tgt, cont_tgt, _ = (t.to(self.device) for t in batch_items)
            qtarget = None
        
        return cat_input, cont_inp, cat_tgt, cont_tgt, qtarget
    
    def _forward_mlm(
        self,
        cat_input: torch.Tensor,
        cont_inp: torch.Tensor,
        cat_labels: torch.Tensor,
        cont_labels: torch.Tensor,
        qtarget_labels: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Efficient MLM forward pass using pre-masked inputs from data collator."""
        
        if mask is not None:
            # New efficient approach: inputs are already masked, just compute logits
            logits = self.model(cat_input, cont_inp, mode="mlm")  # (B, L, total_vocab_size)
            
            # Compute loss using the efficient masked loss function
            if qtarget_labels is not None:
                # Use quantized targets for continuous features
                loss = self._compute_efficient_mlm_loss(
                    logits, cat_labels, qtarget_labels, mask
                )
            else:
                # Fallback to categorical labels only
                loss = self._compute_efficient_mlm_loss_cat_only(
                    logits, cat_labels, mask
                )
            
            # Count masked elements for logging
            if len(mask.shape) == 3:  # Field-level mask
                masked_elements = mask.sum().item()
            else:  # Row-level mask  
                masked_elements = mask.sum().item() * (len(self.cat_features) + len(self.cont_features))
            
            return loss, {
                "mask_prob": self.args.mask_prob,
                "masked_elements": masked_elements,
                "efficient_masking": True
            }
        else:
            # Fallback to old inefficient approach for compatibility
            # Calculate total number of features
            num_cat_features = len(self.cat_features)
            num_cont_features = len(self.cont_features)
            total_features = num_cat_features + num_cont_features
            
            # Create field-level and row-level mask
            mask = create_field_and_row_mask(
                batch_size=cat_input.shape[0], 
                seq_len=cat_input.shape[1],
                num_features=total_features,
                field_mask_prob=self.args.mask_prob,  # 15% field-level masking
                row_mask_prob=0.10,  # 10% row-level masking
                device=self.device
            )
            
            # Apply masking to inputs
            masked_cat_input = apply_field_level_categorical_masking(
                cat_input, mask, num_cat_features
            )
            masked_cont_input = apply_field_level_continuous_masking(
                cont_inp, mask, num_cat_features
            )
            
            # Forward pass - use standard MLM mode (not sparse) for field-level masking
            logits = self.model(masked_cat_input, masked_cont_input, mode="mlm")  # (B, L, total_vocab_size)
            
            # Compute loss using field-level mask
            loss = compute_field_level_mlm_loss(
                logits=logits,
                targets_cat=cat_labels,
                targets_cont=qtarget_labels,
                mask=mask,
                vocab_sizes=self.vocab_sizes,
                cont_vocab_sizes=self.config.cont_vocab_sizes or {},
                cont_features=self.cont_features,
                criterion=self.criterion_cat
            )
            
            # Count masked elements for logging
            masked_elements = mask.sum().item()
            
            return loss, {
                "mask_prob": self.args.mask_prob,
                "masked_elements": masked_elements,
                "field_level_masking": True,
                "efficient_masking": False
            }

    def _compute_efficient_mlm_loss(
        self,
        logits: torch.Tensor,
        cat_labels: torch.Tensor,
        qtarget_labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MLM loss efficiently using pre-computed labels.
        Only computes loss on masked positions (labels != -100).
        """
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        num_losses = 0
        
        # Get vocabulary offsets for each feature
        vocab_offsets = [0]
        for vocab_size in self.vocab_sizes:
            vocab_offsets.append(vocab_offsets[-1] + vocab_size)
        
        # Add continuous vocabulary sizes
        cont_vocab_sizes = self.config.cont_vocab_sizes or {}
        for feature in self.cont_features:
            vocab_size = cont_vocab_sizes.get(feature, 100)  # Default to 100 bins
            vocab_offsets.append(vocab_offsets[-1] + vocab_size)
        
        # Compute loss for categorical features
        for i, feature in enumerate(self.cat_features):
            start_idx = vocab_offsets[i]
            end_idx = vocab_offsets[i + 1]
            
            feature_logits = logits[:, :, start_idx:end_idx]  # (B, L, V_i)
            feature_labels = cat_labels[:, :, i]  # (B, L)
            
            # Only compute loss where labels != -100
            valid_mask = feature_labels != -100
            if valid_mask.any():
                valid_logits = feature_logits[valid_mask]  # (N, V_i)
                valid_labels = feature_labels[valid_mask]  # (N,)
                
                loss = F.cross_entropy(valid_logits, valid_labels)
                total_loss = total_loss + loss
                num_losses = num_losses + 1
        
        # Compute loss for continuous features (quantized)
        for i, feature in enumerate(self.cont_features):
            cat_idx = len(self.cat_features) + i
            start_idx = vocab_offsets[cat_idx]
            end_idx = vocab_offsets[cat_idx + 1]
            
            feature_logits = logits[:, :, start_idx:end_idx]  # (B, L, V_i)
            feature_labels = qtarget_labels[:, :, i]  # (B, L)
            
            # Only compute loss where labels != -100
            valid_mask = feature_labels != -100
            if valid_mask.any():
                valid_logits = feature_logits[valid_mask]  # (N, V_i)
                valid_labels = feature_labels[valid_mask]  # (N,)
                
                loss = F.cross_entropy(valid_logits, valid_labels)
                total_loss = total_loss + loss
                num_losses = num_losses + 1
        
        # Average the losses
        if num_losses > 0:
            return total_loss / num_losses
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

    def _compute_efficient_mlm_loss_cat_only(
        self,
        logits: torch.Tensor,
        cat_labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MLM loss for categorical features only.
        Only computes loss on masked positions (labels != -100).
        """
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        num_losses = 0
        
        # Get vocabulary offsets for each feature
        vocab_offsets = [0]
        for vocab_size in self.vocab_sizes:
            vocab_offsets.append(vocab_offsets[-1] + vocab_size)
        
        # Compute loss for categorical features
        for i, feature in enumerate(self.cat_features):
            start_idx = vocab_offsets[i]
            end_idx = vocab_offsets[i + 1]
            
            feature_logits = logits[:, :, start_idx:end_idx]  # (B, L, V_i)
            feature_labels = cat_labels[:, :, i]  # (B, L)
            
            # Only compute loss where labels != -100
            valid_mask = feature_labels != -100
            if valid_mask.any():
                valid_logits = feature_logits[valid_mask]  # (N, V_i)
                valid_labels = feature_labels[valid_mask]  # (N,)
                
                loss = F.cross_entropy(valid_logits, valid_labels)
                total_loss = total_loss + loss
                num_losses = num_losses + 1
        
        # Average the losses
        if num_losses > 0:
            return total_loss / num_losses
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _forward_ar(
        self,
        cat_input: torch.Tensor,
        cont_inp: torch.Tensor,
        cat_tgt: torch.Tensor,
        cont_tgt: torch.Tensor,
        qtarget: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """AR forward pass."""
        logits = self.model(cat_input, cont_inp, mode="ar")
        
        # Get all feature vocab sizes (categorical + quantized continuous)
        all_vocab_sizes = list(self.vocab_sizes)  # categorical vocab sizes
        if self.config.cont_vocab_sizes:
            all_vocab_sizes.extend(list(self.config.cont_vocab_sizes.values()))
        
        # Get all targets (categorical + quantized continuous)
        if qtarget is not None:
            all_targets = torch.cat([cat_tgt, qtarget], dim=1)  # (B, total_features)
        else:
            all_targets = cat_tgt  # fallback if no quantized targets
        
        # Compute loss for all features uniformly
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        start_idx = 0
        
        for i, vocab_len in enumerate(all_vocab_sizes):
            total_loss = total_loss + self.criterion_cat(
                logits[:, start_idx:start_idx+vocab_len], all_targets[:, i]
            )
            start_idx = start_idx + vocab_len
        
        # Average loss across all features
        loss = total_loss / len(all_vocab_sizes)
        
        return loss, {"num_features": len(all_vocab_sizes)}
        
    def evaluate_model(self, val_loader: DataLoader, show_predictions: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            show_predictions: Whether to show sample predictions
            
        Returns:
            Tuple of (val_loss, feature_accuracies)
        """
        return evaluate(
            self.model, val_loader, self.cat_features, self.vocab_sizes,
            self.cont_features, self.config.cont_vocab_sizes or {},
            self.criterion_cat, self.device, mode=self.args.mode,
            show_predictions=show_predictions,
            encoders=self.encoders,
            qparams=self.qparams
        )
    
    def save_checkpoint(self, ckpt_path: Path, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            ckpt_path: Path to save checkpoint
            epoch: Current epoch
        """
        save_ckpt(
            self.model, self.optimizer, epoch, self.best_val_loss,
            ckpt_path, self.cat_features, self.cont_features, self.config
        )
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss - 1e-5:
            self.epochs_without_improvement = 0
            self.best_val_loss = val_loss
            return False
        else:
            self.epochs_without_improvement = self.epochs_without_improvement + 1
            return self.epochs_without_improvement >= self.patience 