"""
MLM masking utilities with optimized loss computation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def create_mlm_mask(
    batch_size: int, 
    seq_len: int, 
    mask_prob: float = 0.15, 
    device: torch.device = torch.device("cpu")
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


def create_field_and_row_mask(
    batch_size: int,
    seq_len: int, 
    num_features: int,
    field_mask_prob: float = 0.15,
    row_mask_prob: float = 0.10,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create a combined field-level and row-level mask for MLM training.
    
    This creates a mask that combines:
    1. Field-level masking: Randomly mask individual fields with field_mask_prob
    2. Row-level masking: Randomly mask entire rows with row_mask_prob
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_features: Total number of features (categorical + continuous)
        field_mask_prob: Probability of masking individual fields (default: 0.15)
        row_mask_prob: Probability of masking entire rows (default: 0.10)
        device: Device to put mask on
        
    Returns:
        Boolean mask tensor of shape (batch_size, seq_len, num_features)
        True indicates positions to mask
    """
    # Initialize mask tensor
    mask = torch.zeros(batch_size, seq_len, num_features, dtype=torch.bool, device=device)
    
    # 1. Field-level masking: randomly mask individual fields
    field_mask = torch.rand(batch_size, seq_len, num_features, device=device) < field_mask_prob
    mask = mask | field_mask
    
    # 2. Row-level masking: randomly mask entire rows
    row_mask = torch.rand(batch_size, seq_len, device=device) < row_mask_prob  # (B, L)
    row_mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, num_features)  # (B, L, F)
    mask = mask | row_mask_expanded
    
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


def apply_field_level_categorical_masking(
    cat_input: torch.Tensor,
    mask: torch.Tensor,
    num_cat_features: int,
    mask_token_id: int = 0
) -> torch.Tensor:
    """
    Apply field-level masking to categorical features.
    
    Args:
        cat_input: Categorical input tensor (B, L, C)
        mask: Boolean mask (B, L, total_features) - we use only the categorical part
        num_cat_features: Number of categorical features
        mask_token_id: Token ID to use for masked positions
        
    Returns:
        Masked categorical input tensor
    """
    cat_masked = cat_input.clone()
    
    # Extract categorical mask (first num_cat_features dimensions)
    cat_mask = mask[:, :, :num_cat_features]  # (B, L, C)
    
    # Apply mask to each categorical feature individually
    for i in range(num_cat_features):
        cat_masked[:, :, i] = torch.where(
            cat_mask[:, :, i],
            torch.full_like(cat_masked[:, :, i], mask_token_id),
            cat_masked[:, :, i]
        )
    
    return cat_masked


def apply_field_level_continuous_masking(
    cont_input: torch.Tensor,
    mask: torch.Tensor,
    num_cat_features: int,
    mask_value: float = 0.0
) -> torch.Tensor:
    """
    Apply field-level masking to continuous features.
    
    Args:
        cont_input: Continuous input tensor (B, L, F)
        mask: Boolean mask (B, L, total_features) - we use only the continuous part
        num_cat_features: Number of categorical features (to offset into mask)
        mask_value: Value to use for masked positions
        
    Returns:
        Masked continuous input tensor
    """
    cont_masked = cont_input.clone()
    num_cont_features = cont_input.size(-1)
    
    # Extract continuous mask (after categorical features)
    cont_mask = mask[:, :, num_cat_features:num_cat_features+num_cont_features]  # (B, L, F)
    
    # Apply mask to each continuous feature individually
    for i in range(num_cont_features):
        cont_masked[:, :, i] = torch.where(
            cont_mask[:, :, i],
            torch.full_like(cont_masked[:, :, i], mask_value),
            cont_masked[:, :, i]
        )
    
    return cont_masked


def compute_fast_mlm_loss(
    logits: torch.Tensor,  # (B, L, total_vocab_size)
    targets_cat: torch.Tensor,  # (B, L, num_cat_features)
    targets_cont: Optional[torch.Tensor],  # (B, L, num_cont_features) - quantized targets
    mask: torch.Tensor,  # (B, L)
    vocab_sizes: List[int],
    cont_vocab_sizes: Dict[str, int],
    cont_features: List[str],
    criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Fast MLM loss computation using vectorized operations.
    Avoids expensive indexing and processes all features efficiently.
    
    Args:
        logits: Model predictions (B, L, total_vocab_size)
        targets_cat: Categorical targets (B, L, num_cat_features)
        targets_cont: Quantized continuous targets (B, L, num_cont_features)
        mask: Boolean mask (B, L) - True for masked positions
        vocab_sizes: Vocabulary sizes for categorical features
        cont_vocab_sizes: Vocabulary sizes for continuous features
        cont_features: Names of continuous features
        criterion: Loss function (should be CrossEntropyLoss)
        
    Returns:
        Average loss across all masked positions and features
    """
    B, L = mask.shape
    device = logits.device
    
    # Early exit if no masked positions
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create expanded mask for loss computation: (B*L,)
    flat_mask = mask.view(-1)  # (B*L,)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_features = 0
    start_idx = 0
    
    # Process categorical features
    for i, vocab_size in enumerate(vocab_sizes):
        # Extract feature logits and targets
        feature_logits = logits[:, :, start_idx:start_idx+vocab_size]  # (B, L, vocab_size)
        feature_targets = targets_cat[:, :, i]  # (B, L)
        
        # Flatten to (B*L, vocab_size) and (B*L,)
        flat_logits = feature_logits.view(-1, vocab_size)  # (B*L, vocab_size)
        flat_targets = feature_targets.view(-1)  # (B*L,)
        
        # Use mask to weight the loss (much faster than indexing)
        # CrossEntropyLoss with reduction='none' gives per-sample loss
        sample_losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')  # (B*L,)
        
        # Apply mask and sum only masked losses
        masked_loss = (sample_losses * flat_mask.float()).sum()
        total_loss = total_loss + masked_loss
        num_features += 1
        start_idx += vocab_size
    
    # Process quantized continuous features
    if targets_cont is not None and cont_vocab_sizes:
        for i, feat in enumerate(cont_features):
            if feat in cont_vocab_sizes:
                vocab_size = cont_vocab_sizes[feat]
                
                feature_logits = logits[:, :, start_idx:start_idx+vocab_size]  # (B, L, vocab_size)
                feature_targets = targets_cont[:, :, i]  # (B, L)
                
                # Flatten
                flat_logits = feature_logits.view(-1, vocab_size)
                flat_targets = feature_targets.view(-1)
                
                # Compute masked loss
                sample_losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                masked_loss = (sample_losses * flat_mask.float()).sum()
                total_loss = total_loss + masked_loss
                num_features += 1
                start_idx += vocab_size
    
    # Average across masked elements
    total_masked_elements = flat_mask.sum() * num_features
    if total_masked_elements > 0:
        return total_loss / total_masked_elements
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


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
    DEPRECATED: Legacy MLM loss function for backward compatibility.
    Use compute_fast_mlm_loss for better performance.
    """
    # This function is kept for compatibility but should not be used
    # as it has the old separate cat/cont head structure
    raise NotImplementedError("Use compute_fast_mlm_loss with unified heads instead")


def compute_sparse_mlm_loss(
    logits: torch.Tensor,                    # (num_masked, total_vocab_size)
    masked_positions: torch.Tensor,          # (num_masked, 2) [batch_idx, seq_idx]
    targets_cat: torch.Tensor,               # (B, L, num_cat_features)
    targets_cont: Optional[torch.Tensor],    # (B, L, num_cont_features)
    vocab_sizes: List[int],
    cont_vocab_sizes: Dict[str, int],
    cont_features: List[str],
    criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Efficient sparse MLM loss computation - only compute loss for masked positions.
    
    Args:
        logits: Sparse model predictions (num_masked, total_vocab_size)
        masked_positions: Positions of masked tokens (num_masked, 2) [batch_idx, seq_idx]
        targets_cat: Categorical targets (B, L, num_cat_features)
        targets_cont: Quantized continuous targets (B, L, num_cont_features)
        vocab_sizes: Vocabulary sizes for categorical features
        cont_vocab_sizes: Vocabulary sizes for continuous features
        cont_features: Names of continuous features
        criterion: Loss function (should be CrossEntropyLoss)
        
    Returns:
        Average loss across all masked positions and features
    """
    device = logits.device
    num_masked = len(masked_positions)
    
    # Early exit if no masked positions
    if num_masked == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Extract targets only for masked positions
    batch_indices = masked_positions[:, 0]  # (num_masked,)
    seq_indices = masked_positions[:, 1]    # (num_masked,)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_features = 0
    start_idx = 0
    
    # Process categorical features
    for i, vocab_size in enumerate(vocab_sizes):
        # Extract logits for this feature: (num_masked, vocab_size)
        feature_logits = logits[:, start_idx:start_idx+vocab_size]
        
        # Extract targets for masked positions: (num_masked,)
        feature_targets = targets_cat[batch_indices, seq_indices, i]
        
        # Compute loss directly (no masking needed - all positions are relevant)
        loss = criterion(feature_logits, feature_targets)
        total_loss = total_loss + loss
        num_features += 1
        start_idx += vocab_size
    
    # Process quantized continuous features
    if targets_cont is not None and cont_vocab_sizes:
        for i, feat in enumerate(cont_features):
            if feat in cont_vocab_sizes:
                vocab_size = cont_vocab_sizes[feat]
                
                # Extract logits for this feature: (num_masked, vocab_size)
                feature_logits = logits[:, start_idx:start_idx+vocab_size]
                
                # Extract targets for masked positions: (num_masked,)
                feature_targets = targets_cont[batch_indices, seq_indices, i]
                
                # Compute loss
                loss = criterion(feature_logits, feature_targets)
                total_loss = total_loss + loss
                num_features += 1
                start_idx += vocab_size
    
    # Average across all features (each masked position contributes once per feature)
    if num_features > 0:
        return total_loss / num_features
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_field_level_mlm_loss(
    logits: torch.Tensor,                    # (B, L, total_vocab_size) or (num_masked, total_vocab_size)
    targets_cat: torch.Tensor,               # (B, L, num_cat_features)
    targets_cont: Optional[torch.Tensor],    # (B, L, num_cont_features)
    mask: torch.Tensor,                      # (B, L, total_features) - 3D field-level mask
    vocab_sizes: List[int],
    cont_vocab_sizes: Dict[str, int],
    cont_features: List[str],
    criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Compute MLM loss for field-level masking.
    Only computes loss for features that were actually masked.
    
    Args:
        logits: Model predictions (B, L, total_vocab_size) or (num_masked, total_vocab_size)
        targets_cat: Categorical targets (B, L, num_cat_features)
        targets_cont: Quantized continuous targets (B, L, num_cont_features)
        mask: 3D field-level mask (B, L, total_features)
        vocab_sizes: Vocabulary sizes for categorical features
        cont_vocab_sizes: Vocabulary sizes for continuous features
        cont_features: Names of continuous features
        criterion: Loss function (should be CrossEntropyLoss)
        
    Returns:
        Average loss across all masked features
    """
    device = logits.device
    B, L, total_features = mask.shape
    
    # Check if any features are masked
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_masked_features = 0
    start_idx = 0
    
    # Process categorical features
    for i, vocab_size in enumerate(vocab_sizes):
        # Extract mask for this categorical feature
        feature_mask = mask[:, :, i]  # (B, L)
        
        # Check if this feature has any masked positions
        if feature_mask.sum() == 0:
            start_idx += vocab_size
            continue
            
        # Extract logits for this feature: (B, L, vocab_size)
        feature_logits = logits[:, :, start_idx:start_idx+vocab_size]
        
        # Extract targets for this feature: (B, L)
        feature_targets = targets_cat[:, :, i]
        
        # Flatten and extract only masked positions
        flat_logits = feature_logits.view(-1, vocab_size)  # (B*L, vocab_size)
        flat_targets = feature_targets.view(-1)  # (B*L,)
        flat_mask = feature_mask.view(-1)  # (B*L,)
        
        # Extract only masked positions
        masked_logits = flat_logits[flat_mask]  # (num_masked_for_this_feature, vocab_size)
        masked_targets = flat_targets[flat_mask]  # (num_masked_for_this_feature,)
        
        if len(masked_logits) > 0:
            # Compute loss only for masked positions
            loss = criterion(masked_logits, masked_targets)
            total_loss = total_loss + loss
            num_masked_features += 1
        
        start_idx += vocab_size
    
    # Process quantized continuous features
    if targets_cont is not None and cont_vocab_sizes:
        for i, feat in enumerate(cont_features):
            if feat in cont_vocab_sizes:
                vocab_size = cont_vocab_sizes[feat]
                feature_idx = len(vocab_sizes) + i  # Index in the mask tensor
                
                # Extract mask for this continuous feature
                feature_mask = mask[:, :, feature_idx]  # (B, L)
                
                # Check if this feature has any masked positions
                if feature_mask.sum() == 0:
                    start_idx += vocab_size
                    continue
                
                # Extract logits for this feature: (B, L, vocab_size)
                feature_logits = logits[:, :, start_idx:start_idx+vocab_size]
                
                # Extract targets for this feature: (B, L)
                feature_targets = targets_cont[:, :, i]
                
                # Flatten and extract only masked positions
                flat_logits = feature_logits.view(-1, vocab_size)  # (B*L, vocab_size)
                flat_targets = feature_targets.view(-1)  # (B*L,)
                flat_mask = feature_mask.view(-1)  # (B*L,)
                
                # Extract only masked positions
                masked_logits = flat_logits[flat_mask]  # (num_masked_for_this_feature, vocab_size)
                masked_targets = flat_targets[flat_mask]  # (num_masked_for_this_feature,)
                
                if len(masked_logits) > 0:
                    # Compute loss only for masked positions
                    loss = criterion(masked_logits, masked_targets)
                    total_loss = total_loss + loss
                    num_masked_features += 1
                
                start_idx += vocab_size
    
    # Average across all masked features
    if num_masked_features > 0:
        return total_loss / num_masked_features
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)