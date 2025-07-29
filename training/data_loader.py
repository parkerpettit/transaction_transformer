"""
Data loading utilities for training.
"""
import torch
from pathlib import Path
from typing import Tuple, Any, List
from torch.utils.data import DataLoader
from data.dataset import TxnDataset, collate_fn
from training.data_collator import create_mlm_collator
from configs.paths import ProjectPaths


def load_processed_data(paths: ProjectPaths, mode: str = "pretrain") -> Tuple[Any, ...]:
    """
    Load processed data from the specified paths.
    
    Args:
        paths: ProjectPaths object containing data paths
        mode: Training mode ("pretrain" or "finetune")
        
    Returns:
        Tuple of (train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams)
    """
    if mode == "pretrain":
        data_path = paths.pretrain_data_path
    else:
        data_path = paths.finetune_data_path
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    print(f"Loading processed data from {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    if len(data) == 7:
        # Legacy format without quantization parameters
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler = data
        qparams = None
    elif len(data) == 8:
        # New format with quantization parameters
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams = data
    else:
        raise ValueError(f"Unexpected data format with {len(data)} elements")
    
    print(f"Loaded data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    print(f"Categorical features: {cat_features}")
    print(f"Continuous features: {cont_features}")
    
    return train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams


def create_dataloaders(
    train_df: Any,
    val_df: Any,
    cat_features: List[str],
    cont_features: List[str],
    batch_size: int,
    window: int,
    stride: int,
    mode: str = "ar",
    mask_prob: float = 0.15,
    masking_mode: str = "field"
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        cat_features: List of categorical feature names
        cont_features: List of continuous feature names
        batch_size: Batch size
        window: Sequence length
        stride: Stride between sequences
        mode: Training mode ("ar", "masked", "mlm")
        mask_prob: Masking probability for MLM
        masking_mode: Masking mode ("field", "row", "both")
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TxnDataset(
        train_df, 
        cat_features[0],  # group_by column
        cat_features, 
        cont_features,
        window=window,
        stride=stride
    )
    
    val_dataset = TxnDataset(
        val_df,
        cat_features[0],  # group_by column
        cat_features,
        cont_features,
        window=window,
        stride=stride
    )
    
    print(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create data collator for MLM mode
    if mode in ["masked", "mlm"]:
        collator = create_mlm_collator(
            mask_prob=mask_prob,
            mode=masking_mode
        )
    else:
        collator = collate_fn
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def verify_data_exists(paths: ProjectPaths, mode: str = "pretrain") -> bool:
    """
    Verify that processed data exists for the specified mode.
    
    Args:
        paths: ProjectPaths object containing data paths
        mode: Training mode ("pretrain" or "finetune")
        
    Returns:
        True if data exists, False otherwise
    """
    if mode == "pretrain":
        data_path = paths.pretrain_data_path
    else:
        data_path = paths.finetune_data_path
    
    return data_path.exists() 