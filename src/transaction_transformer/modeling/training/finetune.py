"""
Finetuning script for fraud detection model.

Supports both MLM and AR finetuning modes.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
import time
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import FraudDetectionModel
from transaction_transformer.modeling.training.trainers.finetune_trainer import FinetuneTrainer
from transaction_transformer.modeling.training.base.checkpoint_manager import CheckpointManager

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import FinetuneCollator
import pandas as pd
from transaction_transformer.utils.wandb_utils import init_wandb, download_artifact

logger = logging.getLogger(__name__)

def _setup_logging(job_name: str = "finetune") -> Path:
    log_dir = Path("logs") / job_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{int(time.time())}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)
    return log_path

def create_datasets(
    df: pd.DataFrame,
    config: Config,
    schema: FieldSchema,
    validation: bool = False,
) -> TxnDataset:
    """Create dataset with config-driven parameters."""
    return TxnDataset(
        df=df,
        group_by=config.model.data.group_by,
        schema=schema,
        window=config.model.data.window,
        stride=config.model.data.stride,
        include_all_fraud=config.model.data.include_all_fraud if not validation else False,
    )

def finetune(
    model: FraudDetectionModel,
    train_dataset: TxnDataset,
    val_dataset: TxnDataset,
    schema: FieldSchema,
    config: Config,
    device: torch.device,
) -> FinetuneTrainer:
    """Train model using AR finetuning."""
    
    # Create collators
    
    train_collator = FinetuneCollator(schema=schema)
    val_collator = FinetuneCollator(schema=schema)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=config.model.training.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=config.model.training.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.training.learning_rate,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.model.training.total_epochs
    )
    

    
   
    # Create trainer
    trainer = FinetuneTrainer(
        model=model,
        schema=schema,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore
    )
    return trainer

def main():
    """Main finetuning function."""
    
    # Load configuration
    config_manager = ConfigManager(config_path="finetune.yaml")
    config = config_manager.config
    print("Configuration loaded")
    
    
    # Init logging & W&B
    log_file = _setup_logging("finetune")
    logger.info("Starting finetune job")
    run = init_wandb(config, job_type="finetune", tags=[config.model.training.model_type])
    if wandb.run is not None:
        logger.info("W&B run initialized | entity=%s project=%s run_id=%s", wandb.run.entity, wandb.run.project, wandb.run.id)

    # Load preprocessed data: artifact-first to data/processed, fallback to local
    processed_dir = Path(config.model.data.preprocessed_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not config.model.data.use_local_inputs and config.model.data.preprocessed_artifact_name and wandb.run is not None:
        ref = f"{config.model.data.preprocessed_artifact_name}:latest"
        logger.info("Loading preprocessed data from artifact via use_artifact: %s", ref)
        t0 = time.time()
        adir = Path(wandb.run.use_artifact(ref).download(root=str(processed_dir)))
        logger.info("Artifact downloaded and loaded in %.2fs", time.time() - t0)
        train_df = pd.read_parquet(adir / "train.parquet")
        val_df = pd.read_parquet(adir / "val.parquet")
        test_df = pd.read_parquet(adir / "test.parquet")
        ft_schema_obj = torch.load(adir / "schema.pt", map_location="cpu", weights_only=False)
    else:
        # Local fallback
        torch_bundle = Path(config.model.data.preprocessed_path)
        if torch_bundle.exists():
            logger.info("Loading preprocessed data from local torch bundle: %s", torch_bundle)
            train_df, val_df, test_df, ft_schema_obj = torch.load(str(torch_bundle), weights_only=False)
        else:
            logger.info("Loading preprocessed data from local parquet in %s", processed_dir)
            train_df = pd.read_parquet(processed_dir / "train.parquet")
            val_df = pd.read_parquet(processed_dir / "val.parquet")
            test_df = pd.read_parquet(processed_dir / "test.parquet")
            ft_schema_obj = torch.load(processed_dir / "schema.pt", map_location="cpu", weights_only=False)
    train_ds = create_datasets(train_df, config, ft_schema_obj, validation=False)
    val_ds = create_datasets(val_df, config, ft_schema_obj, validation=True)
    # Dataset class balance
    def _label_stats(df: pd.DataFrame) -> tuple[int,int,float]:
        pos = int((df["is_fraud"] == 1).sum())
        neg = int((df["is_fraud"] == 0).sum())
        ratio = pos / max(1, pos+neg)
        return pos, neg, ratio
    tr_pos, tr_neg, tr_ratio = _label_stats(train_df)
    va_pos, va_neg, va_ratio = _label_stats(val_df)
    logger.info("Dataset sizes | train=%d (pos=%d, neg=%d, ratio=%.4f) val=%d (pos=%d, neg=%d, ratio=%.4f)", len(train_ds), tr_pos, tr_neg, tr_ratio, len(val_ds), va_pos, va_neg, va_ratio)
    
    # Create model
    model = FraudDetectionModel(config=config.model, schema=ft_schema_obj)
    # Snapshot initial backbone weights for verification after load
    def _clone_backbone_state_dict(m: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    def _delta_norm(sd0: dict[str, torch.Tensor], sd1: dict[str, torch.Tensor]) -> float:
        total = 0.0
        for k, t0 in sd0.items():
            t1 = sd1.get(k)
            if t1 is None:
                continue
            d = (t0.float() - t1.detach().cpu().float()).norm().item()
            total += d
        return float(total)
    initial_backbone_sd = _clone_backbone_state_dict(model.backbone)

    # Finetune init logic: resume or init from pretrained-backbone
    cm = CheckpointManager(config.model.finetune_checkpoint_dir, stage="finetune")
    if config.model.training.resume:
        loaded_resume = False
        # Prefer W&B latest finetuned-model
        if config.metrics.wandb and not config.model.data.use_local_inputs and wandb.run is not None:
            try:
                mode_suffix = config.model.training.model_type
                art_name = f"finetuned-model-{mode_suffix}:latest"
                logger.info("Resuming finetune from W&B artifact: %s", art_name)
                art = wandb.run.use_artifact(art_name)
                adir = Path(art.download())
                b_payload = torch.load(str(adir / "backbone.pt"), map_location="cpu", weights_only=False)
                h_payload = torch.load(str(adir / "clf_head.pt"), map_location="cpu", weights_only=False)
                model.backbone.load_state_dict(b_payload["state_dict"], strict=True)
                getattr(model, "head").load_state_dict(h_payload["state_dict"], strict=True)
                loaded_resume = True
            except Exception:
                logger.exception("Failed to load %s from W&B; trying local resume exports", art_name)
        if not loaded_resume:
            # Local resume from last exports if present
            b_local = Path(config.model.finetune_checkpoint_dir) / "backbone_last.pt"
            h_local = Path(config.model.finetune_checkpoint_dir) / "clf_head_last.pt"
            if b_local.exists() and h_local.exists():
                logger.info("Resuming finetune from local exports: %s, %s", b_local, h_local)
                b_payload = torch.load(str(b_local), map_location="cpu", weights_only=False)
                h_payload = torch.load(str(h_local), map_location="cpu", weights_only=False)
                model.backbone.load_state_dict(b_payload["state_dict"], strict=True)
                getattr(model, "head").load_state_dict(h_payload["state_dict"], strict=True)
                loaded_resume = True
        if not loaded_resume:
            raise FileNotFoundError("resume=true but could not find previous finetuned-model (W&B or local)")
    else:
        if not config.model.training.from_scratch:
            loaded = False
            # Case A: If not using local inputs, pull the canonical best backbone via a fixed artifact name
            if config.metrics.wandb and not config.model.data.use_local_inputs and wandb.run is not None:
                try:
                    mode_suffix = config.model.training.model_type
                    pre_name = f"pretrained-backbone-{mode_suffix}:best"
                    logger.info("Initializing finetuning from W&B artifact: %s", pre_name)
                    art = wandb.run.use_artifact(pre_name)
                    adir = Path(art.download())
                    payload = torch.load(str(adir / "backbone.pt"), map_location="cpu", weights_only=False)
                    model.backbone.load_state_dict(payload["state_dict"], strict=True)
                    loaded = True
                except Exception:
                    logger.exception("Failed to download/load '%s' from W&B; will try local fallback", pre_name)

                if loaded:
                    try:
                        delta = _delta_norm(initial_backbone_sd, model.backbone.state_dict())
                        logger.info("Backbone load delta norm: %.6f", delta)
                        if delta == 0.0:
                            raise RuntimeError("Backbone unchanged after W&B load; verify architecture matches.")
                    except Exception:
                        logger.exception("Backbone verification failed after W&B load")

            # Case B: Local-only path when use_local_inputs is True or W&B unavailable
            if not loaded:
                candidate = None
                if config.model.data.use_local_inputs and config.model.training.pretrained_backbone_path:
                    p = Path(config.model.training.pretrained_backbone_path)
                    if p.exists():
                        candidate = p
                if candidate is None:
                    default_local = Path(config.model.pretrain_checkpoint_dir) / "backbone_best.pt"
                    if default_local.exists():
                        candidate = default_local

                if candidate is not None:
                    logger.info("Initializing finetuning from local backbone: %s", candidate)
                    cm.load_export_backbone(str(candidate), model.backbone)
                    try:
                        delta = _delta_norm(initial_backbone_sd, model.backbone.state_dict())
                        logger.info("Backbone load delta norm: %.6f", delta)
                        if delta == 0.0:
                            raise RuntimeError("Backbone unchanged after local load; verify weights.")
                    except Exception:
                        logger.exception("Backbone verification failed after local load")
                    loaded = True

            if not loaded:
                raise FileNotFoundError(
                    "Could not initialize pretrained backbone. With use_local_inputs=false, we auto-download "
                    "mode-specific 'pretrained-backbone-<ar|mlm>:best' via use_artifact. Otherwise, place a local export at "
                    f"{Path(config.model.pretrain_checkpoint_dir) / 'backbone_best.pt'} or set from_scratch=true."
                )
        else:
            logger.info("Finetuning from scratch (backbone randomly initialized)")
    
    device = torch.device(config.get_device())
    model.to(device)
    
    
    # Select training model_type based on config
    trainer = finetune(model, train_ds, val_ds, ft_schema_obj, config, device)

    # Resume via local checkpoints removed in favor of artifact-based lineage
    
    
    # Check if checkpoint exists and load it
    # checkpoint_path = Path(config.model.finetune_checkpoint_dir) / f"{config.model.training.model_type}_{config.model.mode}_best_model.pt"
    # if checkpoint_path.exists():
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     trainer.checkpoint_manager.load_checkpoint(str(checkpoint_path), trainer.model, trainer.optimizer, trainer.scheduler)
    #     print(f"Resuming from epoch {trainer.current_epoch + 1}")
    # else:
    #     print("No checkpoint found, starting from scratch")
    
    # Expose log file for artifact attachment
    import os
    os.environ["TT_FINETUNE_LOG_FILE"] = str(log_file)

    logger.info("Starting training...")
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info("Training for %d epochs", config.model.training.total_epochs)
    trainer.train(config.model.training.total_epochs)
    # Close run explicitly to ensure uploads complete
    if wandb.run is not None:
        try:
            wandb.save(str(log_file), policy="now")
        except Exception:
            logger.debug("Failed to save finetune log to W&B run", exc_info=True)
        wandb.finish()


if __name__ == "__main__":
    main()