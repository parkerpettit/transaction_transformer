"""
Data utilities for training.
"""

import torch
from typing import Dict, List, Tuple
import numpy as np
from transaction_transformer.data.dataset import TxnDataset
from transaction_transformer.data.preprocessing import FieldSchema


def calculate_positive_weight(dataset: TxnDataset) -> float:
    """
    Calculate optimal positive weight for binary classification based on class distribution.

    Args:
        dataset: The dataset to analyze

    Returns:
        float: Recommended positive weight
    """
    # Count positive and negative samples
    positive_count = 0
    negative_count = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"]
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1

    # Calculate different weight options
    ratio = negative_count / positive_count if positive_count > 0 else 1.0
    sqrt_ratio = np.sqrt(ratio)

    print(f"Dataset statistics:")
    print(f"  Positive samples: {positive_count}")
    print(f"  Negative samples: {negative_count}")
    print(f"  Positive ratio: {positive_count / (positive_count + negative_count):.4f}")
    print(f"  Recommended weights:")
    print(f"    Inverse frequency: {ratio:.2f}")
    print(f"    Sqrt inverse frequency: {sqrt_ratio:.2f}")

    # Return sqrt ratio as default (less aggressive)
    return float(sqrt_ratio)


def calculate_positive_weight_from_labels(labels: torch.Tensor) -> float:
    """
    Calculate optimal positive weight from a tensor of labels.

    Args:
        labels: Tensor of binary labels (0 or 1)

    Returns:
        float: Recommended positive weight
    """
    positive_count = (labels == 1).sum().item()
    negative_count = (labels == 0).sum().item()

    ratio = negative_count / positive_count if positive_count > 0 else 1.0
    sqrt_ratio = np.sqrt(ratio)

    print(f"Label statistics:")
    print(f"  Positive samples: {positive_count}")
    print(f"  Negative samples: {negative_count}")
    print(f"  Positive ratio: {positive_count / (positive_count + negative_count):.4f}")
    print(f"  Recommended weights:")
    print(f"    Inverse frequency: {ratio:.2f}")
    print(f"    Sqrt inverse frequency: {sqrt_ratio:.2f}")

    return float(sqrt_ratio)
