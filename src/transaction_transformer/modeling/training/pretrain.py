"""
Pretraining script for feature prediction transformer.

Supports both MLM and AR pretraining modes.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple
import wandb
from pathlib import Path
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import PretrainingModel
from transaction_transformer.modeling.training.trainers.pretrainer import Pretrainer

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import ARTabCollator, MLMTabCollator
import pandas as pd

def create_datasets(df: pd.DataFrame, config: Config, schema: FieldSchema) -> TxnDataset:
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
    run = wandb.run if wandb.run is not None else wandb.init(job_type="pretrain")
    art = run.use_artifact(preprocessed_ref)
    adir = Path(art.download(root=str(download_root)))
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
    schema: FieldSchema = torch.load(schema_path, map_location="cpu", weights_only=False)
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
        raise ValueError(f"Invalid training model_type: {config.model.training.model_type}")
    
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.model.training.total_epochs)
    

    
   
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

    # Init wandb early for artifact use
    if config.metrics.wandb and wandb.run is None:
        wandb.init(project=config.metrics.wandb_project, name=config.metrics.run_name, config=config.to_dict(), job_type="debug-pretrain")

    # Always use LEGIT data for pretraining
    prefer_legit = True

    processed_dir = Path(config.model.data.preprocessed_path).parent

    # Artifact-first default unless use_local_inputs is True
    if not config.model.data.use_local_inputs and config.model.data.preprocessed_artifact_name:
        ref = f"{wandb.run.entity}/{wandb.run.project}/{config.model.data.preprocessed_artifact_name}:latest" if wandb.run else f"{config.model.data.preprocessed_artifact_name}:latest"
        print(f"Loading preprocessed data from artifact: {ref}")
        train_df, val_df, test_df, schema = _load_preprocessed_from_artifact(ref, prefer_legit, processed_dir)
    else:
        # Local override path: expect files already present in data/processed or torch bundle as fallback
        torch_bundle = Path(config.model.data.preprocessed_path)
        if torch_bundle.exists():
            print("Loading preprocessed data from local torch bundle")
            train_df, val_df, test_df, schema = torch.load(str(torch_bundle), weights_only=False)
        else:
            # Try parquet + schema files under data/processed
            print("Loading preprocessed data from local parquet files in data/processed")
            train_df = pd.read_parquet(processed_dir / ("legit_train.parquet" if prefer_legit else "train.parquet"))
            val_df = pd.read_parquet(processed_dir / ("legit_val.parquet" if prefer_legit else "val.parquet"))
            test_df = pd.read_parquet(processed_dir / ("legit_test.parquet" if prefer_legit else "test.parquet"))
            schema: FieldSchema = torch.load(processed_dir / "schema.pt", map_location="cpu", weights_only=False)
    train_ds = create_datasets(train_df, config, schema)
    val_ds = create_datasets(val_df, config, schema)
    
    # Create model
    model = PretrainingModel(config=config.model, schema=schema)
    device = torch.device(config.get_device())
    model.to(device)
    
    # Select training model_type based on config
    trainer = pretrain(model, train_ds, val_ds, schema, config, device)

    print("Starting training...")
    trainer.train(config.model.training.total_epochs)
    # Close the run explicitly for predictable uploads and closure, especially in notebooks
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()