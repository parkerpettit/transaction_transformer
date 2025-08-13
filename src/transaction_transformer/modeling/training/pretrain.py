"""
Pretraining script for feature prediction transformer.

Supports both MLM and AR pretraining modes.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple
import wandb
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import PretrainingModel
from transaction_transformer.modeling.training.trainers.pretrainer import Pretrainer

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import ARTabCollator, MLMTabCollator
import pandas as pd
from transaction_transformer.utils.wandb_utils import init_wandb, download_artifact

logger = logging.getLogger(__name__)


def _setup_logging(job_name: str = "pretrain") -> Path:
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
    df: pd.DataFrame, config: Config, schema: FieldSchema
) -> TxnDataset:
    """Create a dataset from a dataframe and schema."""
    return TxnDataset(
        df=df,
        group_by=config.model.data.group_by,
        schema=schema,
        window=config.model.data.window,
        stride=config.model.data.stride,
        include_all_fraud=config.model.data.include_all_fraud,
    )


def _load_preprocessed_from_artifact(
    preprocessed_ref: str,
    prefer_legit: bool,
    download_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FieldSchema]:
    # Always use lineage if run exists; otherwise, download via public API.
    adir = download_artifact(
        wandb.run, preprocessed_ref, type="dataset", root=str(download_root)
    )
    if prefer_legit:
        train_path = adir / "legit_train.parquet"
        val_path = adir / "legit_val.parquet"
        test_path = adir / "legit_test.parquet"
    else:
        train_path = adir / "train.parquet"
        val_path = adir / "val.parquet"
        test_path = adir / "test.parquet"
    schema_path = adir / "schema.pt"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    schema: FieldSchema = torch.load(
        schema_path, map_location="cpu", weights_only=False
    )
    return train_df, val_df, test_df, schema


def pretrain(
    model: PretrainingModel,
    train_dataset: TxnDataset,
    val_dataset: TxnDataset,
    schema: FieldSchema,
    config: Config,
    device: torch.device,
) -> Pretrainer:
    """Train model using AR pretraining."""

    # Create collators
    if config.model.training.model_type == "ar":
        train_collator = ARTabCollator(config=config.model, schema=schema)
        val_collator = ARTabCollator(config=config.model, schema=schema)
    elif config.model.training.model_type == "mlm":
        train_collator = MLMTabCollator(config=config.model, schema=schema)
        val_collator = MLMTabCollator(config=config.model, schema=schema)
    else:
        raise ValueError(
            f"Invalid training model_type: {config.model.training.model_type}"
        )

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
    trainer = Pretrainer(
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
    """Main pretraining function."""

    # Load configuration
    config_manager = ConfigManager(config_path="pretrain.yaml")
    config = config_manager.config
    print("Configuration loaded")

    log_path = _setup_logging("pretrain")
    os.environ["TT_PRETRAIN_LOG_FILE"] = str(log_path)

    # Init W&B (used to resolve artifacts below)
    run = init_wandb(
        config, job_type="pretrain", tags=[config.model.training.model_type, "pretrain", f"use_amp={config.model.training.use_amp}"], run_id=config.metrics.run_id
    )
    run.use_artifact("preprocessed-card:latest")
    dataset_dir = Path(run.use_artifact("preprocessed-card:latest").download())
    logger.info(f"Dataset directory: {dataset_dir}")

    logger.info("Loading train and val data")
    train_df = pd.read_parquet(dataset_dir / "legit_train.parquet")
    val_df = pd.read_parquet(dataset_dir / "legit_val.parquet")

    # test_df = pd.read_parquet(dataset_dir / "test.parquet")
    schema = torch.load(dataset_dir / "schema.pt", map_location="cpu", weights_only=False)

    logger.info("Creating datasets")
    train_ds = create_datasets(train_df, config, schema)
    val_ds = create_datasets(val_df, config, schema)

    logger.info("Initializing model")
    model = PretrainingModel(config.model, schema)
    
    if config.model.training.resume:
        pretrained_artifact = run.use_artifact(f"{config.model.mode}-{config.model.training.model_type}:latest")
        logger.info("Downloading pretrained artifact: %s", f"{config.model.mode}-{config.model.training.model_type}:latest")
        pretrained_dir = Path(pretrained_artifact.download())

        logger.info("Loading weights from pretrained backbone %s", pretrained_dir / "backbone.pt")
        backbone = torch.load(pretrained_dir / "backbone.pt", map_location="cpu", weights_only=False)
        model.backbone.load_state_dict(backbone["state_dict"], strict=True)

        logger.info("Loading weights from pretrained head %s", pretrained_dir / "head.pt")
        head = torch.load(pretrained_dir / "head.pt", map_location="cpu", weights_only=False)
        model.head.load_state_dict(head["state_dict"], strict=True)
    else:
        logger.info("Starting from scratch")
    
    device = torch.device(config.get_device())
    model.to(device)


    logger.info("Building pretrain trainer")
    trainer = pretrain(model, train_ds, val_ds, schema, config, device)
    
    num_epochs = config.model.training.total_epochs
    logger.info("Pretraining for %d epochs", num_epochs)
    logger.info("Using %s", f"use_amp={config.model.training.use_amp}")
    time_start = time.time()
    trainer.train(num_epochs)
    time_end = time.time()
    logger.info("Pretraining complete in %d seconds", time_end - time_start)
    logger.info("Pretraining complete")

if __name__ == "__main__":
    main()