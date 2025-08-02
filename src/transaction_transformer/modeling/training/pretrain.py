"""
Pretraining script for feature prediction transformer.

Supports both MLM and AR pretraining modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import wandb
from pathlib import Path
from tqdm import tqdm
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import FeaturePredictionModel
from transaction_transformer.modeling.training.trainers import AutoregressiveTrainer

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema
from transaction_transformer.data.collator import ARTabCollator, MLMTabCollator
import pandas as pd

def create_datasets(
    df: pd.DataFrame,
    config: Config,
    schema: FieldSchema
) -> TxnDataset:
    """Create train and validation datasets."""
    
    # Create datasets
    dataset = TxnDataset(
        df=df,
        group_by=config.model.data.group_by,
        schema=schema,
        window=config.model.data.window,
        stride=config.model.data.stride,
        include_all_fraud=config.model.data.include_all_fraud
    )
    
   
    
    return dataset

def train_ar(
    model: FeaturePredictionModel,
    train_dataset: TxnDataset,
    val_dataset: TxnDataset,
    schema: FieldSchema,
    config: Config,
    device: torch.device,
    wandb_run: Optional[Any] = None
) -> AutoregressiveTrainer:
    """Train model using AR pretraining."""
    
    # Create collators
    train_collator = ARTabCollator(config=config.model, schema=schema)
    val_collator = ARTabCollator(config=config.model, schema=schema)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=config.model.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.training.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=config.model.training.num_workers
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
    trainer = AutoregressiveTrainer(
        model=model,
        schema=schema,
        config=config.model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore
        wandb_run=wandb_run
    )
    
    return trainer


def main():
    """Main pretraining function."""
    
    # Load configuration
    config_manager = ConfigManager(config_path="pretrain.yaml")
    config = config_manager.load_config()

    print("Configuration loaded")
    wandb_run = wandb.init(project=config.metrics.wandb_project, name=config.metrics.run_name, config=config.to_dict(), tags=["ar"])

    
    
    # Load preprocessed data
    train_df, val_df, test_df, schema = torch.load(config.model.data.preprocessed_path, weights_only=False)
    train_ds = create_datasets(train_df, config, schema)
    val_ds = create_datasets(val_df, config, schema)
    
    # Create model
    model = FeaturePredictionModel(config=config.model, schema=schema)
    device = torch.device(config.get_device())
    model.to(device)
    trainer = train_ar(model, train_ds, val_ds, schema, config, device, wandb_run)
    print("Starting training...")
    trainer.train(config.model.training.total_epochs)


if __name__ == "__main__":
    main()