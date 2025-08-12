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

    # Setup logging & W&B
    log_file = _setup_logging("pretrain")
    logger.info("Starting pretraining job")
    run = init_wandb(
        config, job_type="pretrain", tags=[config.model.training.model_type, "pretrain", f"use_amp={config.model.training.use_amp}"]
    )
    if wandb.run is not None:
        logger.info(
            "W&B run initialized | entity=%s project=%s run_id=%s",
            wandb.run.entity,
            wandb.run.project,
            wandb.run.id,
        )

    # Always use LEGIT data for pretraining
    prefer_legit = True

    processed_dir = Path(config.model.data.preprocessed_path).parent

    # Artifact-first default unless use_local_inputs is True. Always use lineage (use_artifact)
    if (
        not config.model.data.use_local_inputs
        and config.model.data.preprocessed_artifact_name
        and wandb.run is not None
    ):
        ref = f"{config.model.data.preprocessed_artifact_name}:latest"
        logger.info("Loading preprocessed data from artifact via use_artifact: %s", ref)
        t0 = time.time()
        adir = wandb.run.use_artifact(ref).download(root=str(processed_dir))
        adir = Path(adir)
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
        pt_schema = torch.load(schema_path, map_location="cpu", weights_only=False)
        logger.info("Artifact downloaded and loaded in %.2fs", time.time() - t0)
    else:
        # Local override path: expect files already present in data/processed or torch bundle as fallback
        torch_bundle = Path(config.model.data.preprocessed_path)
        if torch_bundle.exists():
            logger.info(
                "Loading preprocessed data from local torch bundle: %s", torch_bundle
            )
            train_df, val_df, test_df, pt_schema = torch.load(
                str(torch_bundle), weights_only=False
            )
        else:
            # Try parquet + schema files under data/processed
            logger.info(
                "Loading preprocessed data from local parquet files in %s",
                processed_dir,
            )
            train_df = pd.read_parquet(
                processed_dir
                / ("legit_train.parquet" if prefer_legit else "train.parquet")
            )
            val_df = pd.read_parquet(
                processed_dir / ("legit_val.parquet" if prefer_legit else "val.parquet")
            )
            test_df = pd.read_parquet(
                processed_dir
                / ("legit_test.parquet" if prefer_legit else "test.parquet")
            )
            pt_schema = torch.load(
                processed_dir / "schema.pt", map_location="cpu", weights_only=False
            )
    train_ds = create_datasets(train_df, config, pt_schema)
    val_ds = create_datasets(val_df, config, pt_schema)
    logger.info("Dataset sizes | train=%d val=%d", len(train_ds), len(val_ds))

    # Create model
    model = PretrainingModel(config=config.model, schema=pt_schema)
    device = torch.device(config.get_device())
    model.to(device)
    # Config summary
    logger.info(
        "Config | model_type=%s D(field)=%d M(seq)=%d field_depth=%d heads=%d seq_depth=%d seq_heads=%d",
        config.model.training.model_type,
        config.model.field_transformer.d_model,
        config.model.sequence_transformer.d_model,
        config.model.field_transformer.depth,
        config.model.field_transformer.n_heads,
        config.model.sequence_transformer.depth,
        config.model.sequence_transformer.n_heads,
    )
    logger.info(
        "Training | epochs=%d batch_size=%d lr=%.2e device=%s max_batches_per_epoch=%s",
        config.model.training.total_epochs,
        config.model.training.batch_size,
        config.model.training.learning_rate,
        config.get_device(),
        str(getattr(config.model.training, "max_batches_per_epoch", None)),
    )

    # Select training model_type based on config
    trainer = pretrain(model, train_ds, val_ds, pt_schema, config, device)

    # Expose log file to artifact logger via env var so it can be attached to model artifacts
    os.environ["TT_PRETRAIN_LOG_FILE"] = str(log_file)

    # Optional resume for pretraining: load latest backbone/head from W&B or local
    if config.model.training.resume:
        loaded_resume = False
        if (
            config.metrics.wandb
            and not config.model.data.use_local_inputs
            and wandb.run is not None
        ):
            try:
                mode_suffix = config.model.training.model_type
                pre_latest = f"pretrained-backbone:latest"
                logger.info("Resuming pretrain from W&B artifact: %s", pre_latest)
                art = wandb.run.use_artifact(pre_latest)
                adir = Path(art.download())
                b_payload = torch.load(
                    str(adir / "backbone.pt"), map_location="cpu", weights_only=False
                )
                h_payload = torch.load(
                    str(adir / "pretrain_head.pt"),
                    map_location="cpu",
                    weights_only=False,
                )
                model.backbone.load_state_dict(b_payload["state_dict"], strict=True)
                model.head.load_state_dict(h_payload["state_dict"], strict=True)
                loaded_resume = True
            except Exception:
                logger.exception(
                    "Failed to load %s from W&B; trying local resume exports",
                    pre_latest,
                )
        if not loaded_resume:
            b_local = Path(config.model.pretrain_checkpoint_dir) / "backbone_last.pt"
            h_local = (
                Path(config.model.pretrain_checkpoint_dir) / "pretrain_head_last.pt"
            )
            if b_local.exists() and h_local.exists():
                logger.info(
                    "Resuming pretrain from local exports: %s, %s", b_local, h_local
                )
                b_payload = torch.load(
                    str(b_local), map_location="cpu", weights_only=False
                )
                h_payload = torch.load(
                    str(h_local), map_location="cpu", weights_only=False
                )
                model.backbone.load_state_dict(b_payload["state_dict"], strict=True)
                model.head.load_state_dict(h_payload["state_dict"], strict=True)

    logger.info("Starting training...")
    t_train = time.time()
    trainer.train(config.model.training.total_epochs)
    logger.info("Training complete in %.2fs", time.time() - t_train)

    # Save the log file to the run for convenience
    if wandb.run is not None:
        try:
            wandb.save(str(log_file), policy="now")
        except Exception:
            logger.debug("Failed to save pretrain log to W&B run", exc_info=True)
        # Close the run explicitly for predictable uploads and closure
        wandb.finish()


if __name__ == "__main__":
    main()
