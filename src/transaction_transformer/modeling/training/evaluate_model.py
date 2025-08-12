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
from transaction_transformer.modeling.training.trainers.finetune_trainer import FinetuneTrainer
from transaction_transformer.modeling.training.trainers.evaluater import Evaluater
from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import FinetuneCollator
import pandas as pd
from transaction_transformer.utils.wandb_utils import init_wandb

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
        include_all_fraud=False
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
        pin_memory=True
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
    
    
    # Load preprocessed data
    train_df, val_df, test_df, schema = torch.load(config.model.data.preprocessed_path, weights_only=False)
    val_ds = create_datasets(val_df, config, schema)

    device = torch.device(config.get_device())
    artifact_names = ["finetuned-model-ar:v5", "finetuned-model-mlm:best"]
    
    # Select training model_type based on config
    evaluater = evaluate(artifact_names, val_ds, schema, config, device)
    
    
    # Check if checkpoint exists and load it
    # checkpoint_path = Path(config.model.finetune_checkpoint_dir) / f"{config.model.training.model_type}_{config.model.mode}_best_model.pt"
    # if checkpoint_path.exists():
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     trainer.checkpoint_manager.load_checkpoint(str(checkpoint_path), trainer.model, trainer.optimizer, trainer.scheduler)
    #     print(f"Resuming from epoch {trainer.current_epoch + 1}")
    # else:
    #     print("No checkpoint found, starting from scratch")
    
    print("Starting evaluation...")
    evaluater.evaluate()


if __name__ == "__main__":
    main()