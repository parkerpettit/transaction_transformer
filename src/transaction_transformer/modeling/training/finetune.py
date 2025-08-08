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
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import FraudDetectionModel
from transaction_transformer.modeling.training.trainers.finetune_trainer import FinetuneTrainer

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema
from transaction_transformer.data.collator import FinetuneCollator
import pandas as pd

def create_datasets(
    df: pd.DataFrame,
    config: Config,
    schema: FieldSchema,
    validation: bool = False
) -> TxnDataset:
    """Create train and validation datasets."""
    
    # Create datasets
    dataset = TxnDataset(
        df=df,
        group_by=config.model.data.group_by,
        schema=schema,
        window=config.model.data.window,
        stride=config.model.data.stride,
        include_all_fraud=config.model.data.include_all_fraud if not validation else False
    )
    
   
    
    return dataset

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
    """Main pretraining function."""
    
    # Load configuration
    config_manager = ConfigManager(config_path="finetune.yaml")
    config = config_manager.config
    print("Configuration loaded")
    
    
    # Load preprocessed data
    train_df, val_df, test_df, schema = torch.load(config.model.data.preprocessed_path, weights_only=False)
    train_ds = create_datasets(train_df, config, schema, validation=False)
    val_ds = create_datasets(val_df, config, schema, validation=True)
    
    # Create model
    model = FraudDetectionModel(config=config.model, schema=schema)
    
    # Load pretrained embedding model
    pretrained_checkpoint_path = Path(config.model.pretrain_checkpoint_dir) / f"{config.model.training.model_type}_pretrain_best_model.pt"
    model.load_pretrained_embedding_backbone(str(pretrained_checkpoint_path))
    
    device = torch.device(config.get_device())
    model.to(device)
    
    
    # Select training model_type based on config
    trainer = finetune(model, train_ds, val_ds, schema, config, device)
    
    
    # Check if checkpoint exists and load it
    # checkpoint_path = Path(config.model.finetune_checkpoint_dir) / f"{config.model.training.model_type}_{config.model.mode}_best_model.pt"
    # if checkpoint_path.exists():
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     trainer.checkpoint_manager.load_checkpoint(str(checkpoint_path), trainer.model, trainer.optimizer, trainer.scheduler)
    #     print(f"Resuming from epoch {trainer.current_epoch + 1}")
    # else:
    #     print("No checkpoint found, starting from scratch")
    
    print("Starting training...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print(f"Training for {config.model.training.total_epochs} epochs")
    trainer.train(config.model.training.total_epochs)


if __name__ == "__main__":
    main()