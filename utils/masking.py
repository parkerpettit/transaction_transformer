"""
Masking utilities for MLM (Masked Language Model) training.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
import random


def create_mlm_mask(
    batch_size: int, 
    seq_len: int, 
    mask_prob: float = 0.15,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create a random boolean mask for MLM training.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length  
        mask_prob: Probability of masking each token
        device: Device to put mask on
        
    Returns:
        Boolean mask tensor of shape (batch_size, seq_len)
        True indicates positions to mask
    """
    mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
    return mask


def apply_categorical_masking(
    cat_input: torch.Tensor,
    mask: torch.Tensor, 
    cat_vocab_sizes: Dict[str, int],
    mask_token_id: int = 0
) -> torch.Tensor:
    """
    Apply masking to categorical features.
    
    Args:
        cat_input: Categorical input tensor (B, L, C)
        mask: Boolean mask (B, L) 
        cat_vocab_sizes: Vocabulary sizes for each categorical feature
        mask_token_id: Token ID to use for masked positions
        
    Returns:
        Masked categorical input tensor
    """
    cat_masked = cat_input.clone()
    
    # Apply mask to all categorical features
    for i in range(cat_input.size(-1)):
        cat_masked[:, :, i] = torch.where(
            mask,
            torch.full_like(cat_masked[:, :, i], mask_token_id),
            cat_masked[:, :, i]
        )
    
    return cat_masked


def apply_continuous_masking(
    cont_input: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = 0.0
) -> torch.Tensor:
    """
    Apply masking to continuous features.
    
    Args:
        cont_input: Continuous input tensor (B, L, F)
        mask: Boolean mask (B, L)
        mask_value: Value to use for masked positions
        
    Returns:
        Masked continuous input tensor  
    """
    cont_masked = cont_input.clone()
    
    # Apply mask to all continuous features
    for i in range(cont_input.size(-1)):
        cont_masked[:, :, i] = torch.where(
            mask,
            torch.full_like(cont_masked[:, :, i], mask_value),
            cont_masked[:, :, i]
        )
    
    return cont_masked


def compute_mlm_loss(
    cat_logits: torch.Tensor,
    cont_pred: torch.Tensor,
    cat_targets: torch.Tensor,
    cont_targets: torch.Tensor,
    mask: torch.Tensor,
    cat_vocab_sizes: Dict[str, int],
    cat_features: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MLM loss for both categorical and continuous features.
    
    Args:
        cat_logits: Predicted categorical logits (B, L, total_vocab)
        cont_pred: Predicted continuous values (B, L, F) 
        cat_targets: Target categorical values (B, L, C)
        cont_targets: Target continuous values (B, L, F)
        mask: Boolean mask (B, L) indicating which positions were masked
        cat_vocab_sizes: Vocabulary sizes for categorical features
        cat_features: List of categorical feature names
        
    Returns:
        Tuple of (total_loss, cat_loss, cont_loss)
    """
    device = cat_logits.device
    B, L = mask.shape
    
    # Only compute loss on masked positions
    masked_positions = mask.flatten()  # (B*L,)
    
    if masked_positions.sum() == 0:
        # No masked positions, return zero loss
        return (
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device), 
            torch.tensor(0.0, device=device)
        )
    
    # Categorical loss
    cat_loss = torch.tensor(0.0, device=device)
    if len(cat_features) > 0:
        # Split logits by feature
        logit_splits = list(cat_vocab_sizes.values())
        cat_logit_list = torch.split(cat_logits, logit_splits, dim=-1)
        
        total_cat_loss = 0.0
        valid_features = 0
        
        for i, (feat_name, feat_logits) in enumerate(zip(cat_features, cat_logit_list)):
            feat_targets = cat_targets[:, :, i].flatten()  # (B*L,)
            feat_logits_flat = feat_logits.view(-1, feat_logits.size(-1))  # (B*L, vocab_size)
            
            # Only compute loss on masked positions
            masked_targets = feat_targets[masked_positions]
            masked_logits = feat_logits_flat[masked_positions]
            
            if len(masked_targets) > 0:
                feat_loss = F.cross_entropy(masked_logits, masked_targets, ignore_index=0)
                total_cat_loss += feat_loss
                valid_features += 1
        
        if valid_features > 0:
            cat_loss = total_cat_loss / valid_features
    
    # Continuous loss (MSE)
    cont_loss = torch.tensor(0.0, device=device)
    if cont_targets.size(-1) > 0:
        cont_pred_flat = cont_pred.view(-1, cont_pred.size(-1))  # (B*L, F)
        cont_targets_flat = cont_targets.view(-1, cont_targets.size(-1))  # (B*L, F)
        
        masked_cont_pred = cont_pred_flat[masked_positions]  # (num_masked, F)
        masked_cont_targets = cont_targets_flat[masked_positions]  # (num_masked, F)
        
        if len(masked_cont_pred) > 0:
            cont_loss = F.mse_loss(masked_cont_pred, masked_cont_targets)
    
    # Combine losses
    total_loss = cat_loss + cont_loss
    
    return total_loss, cat_loss, cont_loss