"""
Label smoothing utilities for categorical and quantized continuous features.

Implements:
1. Standard label smoothing for categorical features
2. Neighborhood label smoothing for quantized continuous features
"""

import torch
import torch.nn.functional as F
from typing import Dict, List


def create_categorical_label_smoothing(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1) -> torch.Tensor:
    """
    Create label smoothing for categorical features using equation (3):
    p(v)_l = 1 - epsilon if l = v, else epsilon/(q_j - 1)
    
    Args:
        targets: (B,) tensor of target class indices
        vocab_size: Size of vocabulary (q_j)
        epsilon: Smoothing parameter
        
    Returns:
        (B, vocab_size) tensor of smoothed probability distributions
    """
    batch_size = targets.size(0)
    device = targets.device
    
    # Create one-hot encoding efficiently
    one_hot = torch.zeros(batch_size, vocab_size, device=device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    
    # Apply label smoothing vectorized
    smoothed = one_hot * (1.0 - epsilon) + (1.0 - one_hot) * (epsilon / (vocab_size - 1))
    
    return smoothed


def create_neighborhood_label_smoothing_vectorized(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1, 
                                                  neighborhood_size: int = 5) -> torch.Tensor:
    """
    Vectorized neighborhood label smoothing for quantized continuous features.
    Much faster than the loop-based version.
    """
    batch_size = targets.size(0)
    device = targets.device
    
    # Create range of all possible bins
    all_bins = torch.arange(vocab_size, device=device).unsqueeze(0)  # (1, vocab_size)
    targets_expanded = targets.unsqueeze(1)  # (B, 1)
    
    # Calculate distances from each target to all bins
    distances = torch.abs(all_bins - targets_expanded)  # (B, vocab_size)
    
    # Create mask for neighborhood (distance <= neighborhood_size)
    in_neighborhood = distances <= neighborhood_size  # (B, vocab_size)
    is_target = distances == 0  # (B, vocab_size)
    
    # Initialize smoothed probabilities
    smoothed = torch.zeros(batch_size, vocab_size, device=device)
    
    # Set target probabilities
    smoothed[is_target] = 1.0 - epsilon
    
    # Set neighborhood probabilities (excluding target)
    neighborhood_not_target = in_neighborhood & ~is_target
    smoothed[neighborhood_not_target] = epsilon / 10.0
    
    return smoothed


def compute_smoothed_loss_fast(logits: torch.Tensor, targets: torch.Tensor, 
                              vocab_sizes: List[int], cat_features: List[str], 
                              cont_vocab_sizes: Dict[str, int], cont_features: List[str],
                              epsilon: float = 0.1, neighborhood_size: int = 5) -> torch.Tensor:
    """
    Optimized version of smoothed loss computation.
    """
    device = logits.device
    batch_size = logits.size(0)
    losses = []
    start_idx = 0
    
    # Process categorical features
    for i, vocab_size in enumerate(vocab_sizes):
        feature_logits = logits[:, start_idx:start_idx + vocab_size]
        feature_targets = targets[:, i]
        
        # Fast label smoothing using built-in cross entropy with soft targets
        smoothed_targets = create_categorical_label_smoothing(
            feature_targets, vocab_size, epsilon
        )
        
        # Use more efficient loss computation
        log_probs = F.log_softmax(feature_logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        losses.append(loss)
        
        start_idx += vocab_size
    
    # Process quantized continuous features
    for i, feature_name in enumerate(cont_features):
        if feature_name in cont_vocab_sizes:
            vocab_size = cont_vocab_sizes[feature_name]
            feature_logits = logits[:, start_idx:start_idx + vocab_size]
            feature_targets = targets[:, len(cat_features) + i]
            
            # Use vectorized neighborhood smoothing
            smoothed_targets = create_neighborhood_label_smoothing_vectorized(
                feature_targets, vocab_size, epsilon, neighborhood_size
            )
            
            log_probs = F.log_softmax(feature_logits, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
            losses.append(loss)
            
            start_idx += vocab_size
    
    # Stack and average losses
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


# Keep the old functions for backward compatibility but mark as deprecated
def create_neighborhood_label_smoothing(targets: torch.Tensor, vocab_size: int, epsilon: float = 0.1, 
                                      neighborhood_size: int = 5) -> torch.Tensor:
    """
    DEPRECATED: Use create_neighborhood_label_smoothing_vectorized for better performance.
    """
    return create_neighborhood_label_smoothing_vectorized(targets, vocab_size, epsilon, neighborhood_size)


def compute_smoothed_loss(logits: torch.Tensor, targets: torch.Tensor, 
                         vocab_sizes: List[int], cat_features: List[str], 
                         cont_vocab_sizes: Dict[str, int], cont_features: List[str],
                         epsilon: float = 0.1, neighborhood_size: int = 5) -> torch.Tensor:
    """
    DEPRECATED: Use compute_smoothed_loss_fast for better performance.
    """
    return compute_smoothed_loss_fast(logits, targets, vocab_sizes, cat_features, 
                                    cont_vocab_sizes, cont_features, epsilon, neighborhood_size)


def compute_masked_smoothed_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                vocab_sizes: List[int], cat_features: List[str],
                                cont_vocab_sizes: Dict[str, int], cont_features: List[str],
                                epsilon: float = 0.1, neighborhood_size: int = 5) -> torch.Tensor:
    """
    Compute MLM loss with label smoothing (equation 5 from paper).
    Only computes loss for masked positions.
    """
    B, L = mask.shape
    device = logits.device
    losses = []
    start_idx = 0
    
    # Process categorical features
    for i, vocab_size in enumerate(vocab_sizes):
        feature_logits = logits[:, :, start_idx:start_idx + vocab_size]  # (B, L, vocab_size)
        feature_targets = targets[:, :, i]  # (B, L)
        
        # Only compute loss for masked positions
        masked_positions = mask.flatten()  # (B*L,)
        if masked_positions.sum() == 0:
            continue
            
        # Flatten and select masked positions
        flat_logits = feature_logits.view(-1, vocab_size)[masked_positions]  # (num_masked, vocab_size)
        flat_targets = feature_targets.flatten()[masked_positions]  # (num_masked,)
        
        # Apply standard label smoothing
        smoothed_targets = create_categorical_label_smoothing(
            flat_targets, vocab_size, epsilon
        )
        
        # Compute cross-entropy with smoothed targets
        log_probs = F.log_softmax(flat_logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        losses.append(loss)
        
        start_idx += vocab_size
    
    # Process quantized continuous features
    for i, feature_name in enumerate(cont_features):
        if feature_name in cont_vocab_sizes:
            vocab_size = cont_vocab_sizes[feature_name]
            feature_logits = logits[:, :, start_idx:start_idx + vocab_size]  # (B, L, vocab_size)
            feature_targets = targets[:, :, len(cat_features) + i]  # (B, L)
            
            # Only compute loss for masked positions
            masked_positions = mask.flatten()  # (B*L,)
            if masked_positions.sum() == 0:
                continue
                
            # Flatten and select masked positions
            flat_logits = feature_logits.view(-1, vocab_size)[masked_positions]  # (num_masked, vocab_size)
            flat_targets = feature_targets.flatten()[masked_positions]  # (num_masked,)
            
            # Apply vectorized neighborhood label smoothing
            smoothed_targets = create_neighborhood_label_smoothing_vectorized(
                flat_targets, vocab_size, epsilon, neighborhood_size
            )
            
            # Compute cross-entropy with smoothed targets
            log_probs = F.log_softmax(flat_logits, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
            losses.append(loss)
            
            start_idx += vocab_size
    
    # Average across all losses
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True) 