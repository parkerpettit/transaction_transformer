"""
Finetuning script for fraud detection model.

Supports both MLM and AR finetuning modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import wandb
from pathlib import Path
from tqdm import tqdm
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import FraudDetectionModel
from transaction_transformer.modeling.training.trainers.finetune_trainer import (
    FinetuneTrainer,
)
from transaction_transformer.modeling.training.trainers.evaluater import Evaluater
from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import FinetuneCollator
import pandas as pd
from transaction_transformer.utils.wandb_utils import init_wandb
import logging
import time


def create_datasets(
    df: pd.DataFrame,
    config: Config,
    schema: FieldSchema,
) -> TxnDataset:
    """Create dataset."""

    # Create datasets
    dataset = TxnDataset(
        df=df,
        group_by=config.model.data.group_by,
        schema=schema,
        window=config.model.data.window,
        stride=1,
        include_all_fraud=False,
    )

    return dataset


def evaluate(
    model_paths: List[str],
    val_dataset: TxnDataset,
    schema: FieldSchema,
    config: Config,
    device: torch.device,
) -> Evaluater:
    """Evaluate model using AR finetuning."""

    # Create collators

    val_collator = FinetuneCollator(schema=schema)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=config.model.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # Create evaluater
    evaluater = Evaluater(
        model_paths=model_paths,
        schema=schema,
        config=config,
        device=device,
        val_loader=val_loader,
    )
    return evaluater


def main():
    """Main pretraining function."""
    # validate each saved model
    # Load configuration
    config_manager = ConfigManager(config_path="finetune.yaml")
    config = config_manager.config
    print("Configuration loaded")

    # Init W&B (used to resolve artifacts below)
    run = init_wandb(
        config, job_type="evaluate", tags=[config.model.training.model_type, "evaluate", f"use_amp={config.model.training.use_amp}"]
    )

    # Load preprocessed data: artifact-first to data/processed, fallback to local
    processed_dir = Path(config.model.data.preprocessed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if (
        not config.model.data.use_local_inputs
        and config.model.data.preprocessed_artifact_name
        and wandb.run is not None
    ):
        ref = f"{config.model.data.preprocessed_artifact_name}:latest"
        print(f"Loading preprocessed data from artifact via use_artifact: {ref}")
        t0 = time.time()
        adir = Path(wandb.run.use_artifact(ref).download(root=str(processed_dir)))
        print(f"Artifact downloaded and loaded in {time.time() - t0:.2f}s")
        # Expect parquet + schema.pt layout
        val_df = pd.read_parquet(adir / "val.parquet")
        schema = torch.load(adir / "schema.pt", map_location="cpu", weights_only=False)
    else:
        # Local fallback
        torch_bundle = Path(config.model.data.preprocessed_path)
        if torch_bundle.exists():
            print(f"Loading preprocessed data from local torch bundle: {torch_bundle}")
            _train_df, val_df, _test_df, schema = torch.load(
                str(torch_bundle), weights_only=False
            )
        else:
            print(f"Loading preprocessed data from local parquet in {processed_dir}")
            val_df = pd.read_parquet(processed_dir / "val.parquet")
            schema = torch.load(
                processed_dir / "schema.pt", map_location="cpu", weights_only=False
            )

    val_ds = create_datasets(val_df, config, schema)

    device = torch.device(config.get_device())
    artifact_names = [
        "finetuned-model-ar:best",
        "finetuned-model-mlm:best",
    ]

    evaluater = evaluate(artifact_names, val_ds, schema, config, device)
    print("Starting evaluation...")
    evaluater.evaluate()
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
