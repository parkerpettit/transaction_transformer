"""
Finetuning script for fraud detection model.

Supports both MLM and AR finetuning modes.
"""

import os
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
from transaction_transformer.modeling.training.trainers.finetune_trainer import (
    FinetuneTrainer,
)
from transaction_transformer.modeling.training.base.checkpoint_manager import (
    CheckpointManager,
)

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
    ch.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

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
        stride=config.model.data.stride if not validation else 1,
        include_all_fraud=(
            config.model.data.include_all_fraud if not validation else False
        ),
    )


def build_finetune_trainer(
    model: FraudDetectionModel,
    train_dataset: TxnDataset,
    val_dataset: TxnDataset,
    schema: FieldSchema,
    config: Config,
    device: torch.device,
) -> FinetuneTrainer:
    """Returns finetune trainer."""

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
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=config.model.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.training.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.model.training.total_epochs
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

    log_path = _setup_logging("finetune")
    os.environ["TT_FINETUNE_LOG_FILE"] = str(log_path)

    # Init W&B (used to resolve artifacts below)
    run = init_wandb(
        config, job_type="finetune", tags=[config.model.training.model_type, "finetune", f"use_amp={config.model.training.use_amp}"]
    )
    dataset_dir = Path(run.use_artifact("preprocessed-card:latest").download())
    logger.info(f"Dataset directory: {dataset_dir}")

    logger.info("Loading train and val data")
    train_df = pd.read_parquet(dataset_dir / "train.parquet")
    val_df = pd.read_parquet(dataset_dir / "val.parquet")

    # test_df = pd.read_parquet(dataset_dir / "test.parquet")
    schema = torch.load(dataset_dir / "schema.pt", map_location="cpu", weights_only=False)

    logger.info("Creating datasets")
    train_ds = create_datasets(train_df, config, schema)
    val_ds = create_datasets(val_df, config, schema, validation=True)

    logger.info("Initializing model")
    model = FraudDetectionModel(config.model, schema)
    if config.model.training.resume:
        pretrained_artifact = run.use_artifact(f"finetune-{config.model.training.model_type}:latest")
        logger.info("Resuming from finetune artifact: %s", f"finetune-{config.model.training.model_type}:latest")
        pretrained_dir = Path(pretrained_artifact.download())

        logger.info("Loading weights from finetune backbone %s", pretrained_dir / "backbone.pt")
        backbone = torch.load(pretrained_dir / "backbone.pt", map_location="cpu", weights_only=False)
        model.backbone.load_state_dict(backbone["state_dict"], strict=True)

        logger.info("Loading weights from finetune head %s", pretrained_dir / "head.pt")
        head = torch.load(pretrained_dir / "head.pt", map_location="cpu", weights_only=False)
        model.head.load_state_dict(head["state_dict"], strict=True)
    else:
        pretrained_artifact = run.use_artifact(f"pretrain-{config.model.training.model_type}:latest")
        logger.info("Downloading pretrained artifact: %s", f"pretrain-{config.model.training.model_type}:latest")
        pretrained_dir = Path(pretrained_artifact.download())

        logger.info("Loading weights from pretrained backbone %s", pretrained_dir / "backbone.pt")
        backbone = torch.load(pretrained_dir / "backbone.pt", map_location="cpu", weights_only=False)
        model.backbone.load_state_dict(backbone["state_dict"], strict=True)
    
    device = torch.device(config.get_device())
    model.to(device)


    logger.info("Building finetune trainer")
    trainer = build_finetune_trainer(model, train_ds, val_ds, schema, config, device)
    
    num_epochs = config.model.training.total_epochs
    logger.info("Finetuning for %d epochs", num_epochs)
    logger.info("Using %s", f"use_amp={config.model.training.use_amp}")
    time_start = time.time()
    trainer.train(num_epochs)
    time_end = time.time()
    logger.info("Finetuning complete in %d seconds", time_end - time_start)
    logger.info("Finetuning complete")

if __name__ == "__main__":
    main()
