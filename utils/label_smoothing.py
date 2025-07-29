"""
Label smoothing utilities for training.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def create_categorical_label_smoothing(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1) -> torch.Tensor:
    """
    Create smoothed labels for categorical features.
    
    Args:
        targets: Target tensor (B, L) or (B, L, num_features)
        vocab_size: Vocabulary size for this feature
        epsilon: Smoothing factor (0.1 = 10% smoothing)
        
    Returns:
        Smoothed target tensor with same shape as input
    """
    # Create one-hot encoding
    if len(targets.shape) == 2:
        # (B, L) -> (B, L, vocab_size)
        one_hot = F.one_hot(targets, num_classes=vocab_size).float()
    else:
        # (B, L, num_features) -> (B, L, num_features, vocab_size)
        one_hot = F.one_hot(targets, num_classes=vocab_size).float()
    
    # Apply smoothing
    smoothed = one_hot * (1 - epsilon) + epsilon / vocab_size
    
    return smoothed


def create_neighborhood_label_smoothing_vectorized(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1,
                                                neighborhood_size: int = 1) -> torch.Tensor:
    """
    Create neighborhood-based label smoothing for categorical features.
    Distributes probability mass to neighboring tokens.
    
    Args:
        targets: Target tensor (B, L) or (B, L, num_features)
        vocab_size: Vocabulary size for this feature
        epsilon: Smoothing factor
        neighborhood_size: Size of neighborhood around each token
        
    Returns:
        Smoothed target tensor with same shape as input
    """
    # Create one-hot encoding
    if len(targets.shape) == 2:
        # (B, L) -> (B, L, vocab_size)
        one_hot = F.one_hot(targets, num_classes=vocab_size).float()
    else:
        # (B, L, num_features) -> (B, L, num_features, vocab_size)
        one_hot = F.one_hot(targets, num_classes=vocab_size).float()
    
    # Create neighborhood mask
    device = targets.device
    neighborhood_mask = torch.zeros(vocab_size, vocab_size, device=device)
    
    for i in range(vocab_size):
        # Define neighborhood around token i
        start_idx = max(0, i - neighborhood_size)
        end_idx = min(vocab_size, i + neighborhood_size + 1)
        neighborhood_mask[i, start_idx:end_idx] = 1.0
    
    # Normalize neighborhood mask
    neighborhood_mask = neighborhood_mask / neighborhood_mask.sum(dim=1, keepdim=True)
    
    # Apply neighborhood smoothing
    if len(targets.shape) == 2:
        # (B, L, vocab_size) @ (vocab_size, vocab_size) -> (B, L, vocab_size)
        smoothed = torch.matmul(one_hot, neighborhood_mask)
    else:
        # (B, L, num_features, vocab_size) @ (vocab_size, vocab_size) -> (B, L, num_features, vocab_size)
        B, L, num_features, vocab_size = one_hot.shape
        one_hot_reshaped = one_hot.view(B * L * num_features, vocab_size)
        smoothed_reshaped = torch.matmul(one_hot_reshaped, neighborhood_mask)
        smoothed = smoothed_reshaped.view(B, L, num_features, vocab_size)
    
    # Apply epsilon smoothing
    smoothed = smoothed * (1 - epsilon) + epsilon / vocab_size
    
    return smoothed


def compute_smoothed_loss_fast(logits: torch.Tensor, targets: torch.Tensor,
                              vocab_sizes: List[int], cat_features: List[str],
                              epsilon: float = 0.1) -> torch.Tensor:
    """
    Fast label smoothing loss computation for multiple categorical features.
    
    Args:
        logits: Model predictions (B, L, total_vocab_size)
        targets: Target tensor (B, L, num_cat_features)
        vocab_sizes: List of vocabulary sizes for each feature
        cat_features: List of categorical feature names
        epsilon: Smoothing factor
        
    Returns:
        Average smoothed loss across all features
    """
    device = logits.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    start_idx = 0
    
    for i, (feat_name, vocab_size) in enumerate(zip(cat_features, vocab_sizes)):
        # Extract logits and targets for this feature
        feature_logits = logits[:, :, start_idx:start_idx+vocab_size]  # (B, L, vocab_size)
        feature_targets = targets[:, :, i]  # (B, L)
        
        # Create smoothed targets
        smoothed_targets = create_categorical_label_smoothing(
            feature_targets, vocab_size, epsilon
        )  # (B, L, vocab_size)
        
        # Compute KL divergence loss
        log_probs = F.log_softmax(feature_logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        
        total_loss = total_loss + loss
        start_idx += vocab_size
    
    return total_loss / len(cat_features)


def create_neighborhood_label_smoothing(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1,
                                      neighborhood_size: int = 1) -> torch.Tensor:
    """
    Legacy function - redirects to vectorized version.
    """
    return create_neighborhood_label_smoothing_vectorized(targets, vocab_size, epsilon, neighborhood_size) 