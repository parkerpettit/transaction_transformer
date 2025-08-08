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
from transaction_transformer.modeling.training.base.checkpoint_manager import CheckpointManager

from transaction_transformer.config.config import Config, ConfigManager
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.data.collator import FinetuneCollator
import pandas as pd

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
    
    
    # Init wandb for artifact usage
    if config.metrics.wandb and wandb.run is None:
        wandb.init(project=config.metrics.wandb_project, name=config.metrics.run_name, config=config.to_dict(), job_type="debug-finetune")

    # Load preprocessed data: artifact-first to data/processed, fallback to local
    processed_dir = Path(config.model.data.preprocessed_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not config.model.data.use_local_inputs and config.model.data.preprocessed_artifact_name:
        ref = f"{wandb.run.entity}/{wandb.run.project}/{config.model.data.preprocessed_artifact_name}:latest" if wandb.run else f"{config.model.data.preprocessed_artifact_name}:latest"
        pre_art = wandb.run.use_artifact(ref)  # type: ignore[arg-type]
        pre_dir = Path(pre_art.download(root=str(processed_dir)))
        train_df = pd.read_parquet(pre_dir / "train.parquet")
        val_df = pd.read_parquet(pre_dir / "val.parquet")
        test_df = pd.read_parquet(pre_dir / "test.parquet")
        ft_schema_obj = torch.load(pre_dir / "schema.pt", map_location="cpu", weights_only=False)
    else:
        # Local fallback
        torch_bundle = Path(config.model.data.preprocessed_path)
        if torch_bundle.exists():
            train_df, val_df, test_df, ft_schema_obj = torch.load(str(torch_bundle), weights_only=False)
        else:
            train_df = pd.read_parquet(processed_dir / "train.parquet")
            val_df = pd.read_parquet(processed_dir / "val.parquet")
            test_df = pd.read_parquet(processed_dir / "test.parquet")
            ft_schema_obj = torch.load(processed_dir / "schema.pt", map_location="cpu", weights_only=False)
    train_ds = create_datasets(train_df, config, ft_schema_obj, validation=False)
    val_ds = create_datasets(val_df, config, ft_schema_obj, validation=True)
    
    # Create model
    model = FraudDetectionModel(config=config.model, schema=ft_schema_obj)

    # Finetune init logic
    if config.model.training.resume and config.model.training.resume_path:
        print("Resume from local checkpoint is no longer supported; please use W&B artifacts.")
    else:
        if not config.model.training.from_scratch:
            cm = CheckpointManager(config.model.finetune_checkpoint_dir, stage="finetune")
            loaded = False
            # 1) Explicit artifact ref provided
            if config.model.training.pretrained_backbone_artifact:
                print(f"Initializing finetuning from W&B artifact: {config.model.training.pretrained_backbone_artifact}")
                cm.load_backbone_from_artifact(
                    config.model.training.pretrained_backbone_artifact,
                    model.backbone,
                    wandb_run=wandb.run if config.metrics.wandb else None,
                )
                loaded = True
            # 2) Auto-discover latest pretrain artifact in current project
            if not loaded and wandb.run is not None and config.metrics.wandb:
                entity = wandb.run.entity
                project = wandb.run.project
                auto_ref = CheckpointManager.find_latest_stage_artifact_ref(entity, project, stage="pretrain", prefer_alias="best")
                if auto_ref is None:
                    auto_ref = CheckpointManager.find_latest_stage_artifact_ref(entity, project, stage="pretrain", prefer_alias="latest")
                if auto_ref:
                    print(f"Auto-discovered pretrained backbone artifact: {auto_ref}")
                    cm.load_backbone_from_artifact(
                        auto_ref,
                        model.backbone,
                        wandb_run=wandb.run if config.metrics.wandb else None,
                    )
                    loaded = True
            # 3) Local explicit path (if it exists)
            if not loaded and config.model.training.pretrained_backbone_path:
                candidate = Path(config.model.training.pretrained_backbone_path)
                if candidate.exists():
                    print(f"Initializing finetuning from local pretrained backbone: {candidate}")
                    cm.load_export_backbone(str(candidate), model.backbone)
                    loaded = True
            # 4) Local default fallback: data/models/pretrained/backbone_best.pt
            if not loaded:
                default_local = Path(config.model.pretrain_checkpoint_dir) / "backbone_best.pt"
                if default_local.exists():
                    print(f"Initializing finetuning from local default backbone: {default_local}")
                    cm.load_export_backbone(str(default_local), model.backbone)
                    loaded = True
            if not loaded:
                raise FileNotFoundError(
                    "Could not find pretrained backbone. Set training.pretrained_backbone_artifact, "
                    "or ensure a local export exists at training.pretrained_backbone_path or at "
                    f"{Path(config.model.pretrain_checkpoint_dir) / 'backbone_best.pt'}, or set from_scratch=true."
                )
        else:
            print("Finetuning from scratch (backbone randomly initialized)")
    
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
    
    print("Starting training...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print(f"Training for {config.model.training.total_epochs} epochs")
    trainer.train(config.model.training.total_epochs)
    # Close run explicitly to ensure uploads complete
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()