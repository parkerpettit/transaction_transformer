"""
Data loading utilities for pretraining.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Any, Optional

from data.dataset import TxnDataset, collate_fn
from configs.paths import ProjectPaths


def load_processed_data(paths: ProjectPaths, mode: str = "pretrain") -> Tuple[Any, ...]:
    """
    Load processed data from cache.
    
    Args:
        paths: ProjectPaths instance with configured paths
        mode: "pretrain" or "finetune" to determine which dataset to load
        
    Returns:
        Tuple of (train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams)
        
    Raises:
        FileNotFoundError: If processed data doesn't exist
    """
    if mode == "pretrain":
        cache_path = paths.pretrain_data_path
        data_type = "pretraining"
    elif mode == "finetune":
        cache_path = paths.finetune_data_path  
        data_type = "finetuning"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pretrain' or 'finetune'")
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Processed {data_type} data not found at {cache_path}. "
            "Please run data preprocessing first."
        )
    
    print(f"Loading processed {data_type} data from {cache_path}...")
    data = torch.load(cache_path, weights_only=False)
    print(f"Processed {data_type} data loaded.")
    
    return data


def load_processed_data_legacy(data_dir: str) -> Tuple[Any, ...]:
    """
    Legacy data loading function for backward compatibility.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams)
        
    Raises:
        FileNotFoundError: If processed data doesn't exist
    """
    cache_path = Path(data_dir) / "datasets" / "legitimate_transactions_processed.pt"
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {cache_path}. "
            "Please run data preprocessing first."
        )
    
    print("Loading processed legitimate data...")
    data = torch.load(cache_path, weights_only=False)
    print("Processed legitimate data loaded.")
    
    return data


def create_dataloaders(
    train_df: Any,
    val_df: Any, 
    cat_features: List[str],
    cont_features: List[str],
    batch_size: int,
    window: int,
    stride: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        cat_features: List of categorical feature names
        cont_features: List of continuous feature names
        batch_size: Batch size for training
        window: Sequence length (transactions per sample)
        stride: Stride length between windows
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Creating training loader")
    train_loader = DataLoader(
        TxnDataset(train_df, cat_features[0], cat_features, cont_features, window, stride),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Creating validation loader")
    val_loader = DataLoader(
        TxnDataset(val_df, cat_features[0], cat_features, cont_features, window, stride),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def verify_data_exists(paths: ProjectPaths, mode: str = "pretrain") -> bool:
    """
    Verify that processed data exists.
    
    Args:
        paths: ProjectPaths instance with configured paths
        mode: "pretrain" or "finetune" to determine which dataset to check
        
    Returns:
        True if data exists, False otherwise
    """
    if mode == "pretrain":
        return paths.pretrain_data_path.exists()
    elif mode == "finetune":
        return paths.finetune_data_path.exists()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pretrain' or 'finetune'")


def verify_data_exists_legacy(data_dir: str) -> bool:
    """
    Legacy function to verify data exists for backward compatibility.
    
    Args:
        data_dir: Directory to check for processed data
        
    Returns:
        True if data exists, False otherwise
    """
    cache_path = Path(data_dir) / "datasets" / "legitimate_transactions_processed.pt"
    return cache_path.exists() 