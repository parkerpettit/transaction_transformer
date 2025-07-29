"""
Efficient MLM data collator for tabular data following BERT's approach.
Applies masking during batch formation, not during forward pass.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Any
from data.dataset import collate_fn


class TabularDataCollatorForMLM:
    """
    Data collator for Masked Language Modeling (MLM) on tabular data.
    
    Follows BERT's efficient approach by applying masking during batch formation
    rather than during the forward pass. Supports both field-level and row-level masking.
    
    For tabular data with categorical and continuous features where continuous
    features are quantized into categorical bins, we can use the same masking
    approach for both types.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        field_mask_prob: Optional[float] = None,
        row_mask_prob: float = 0.10,
        mask_token_id: int = 0,
        mask_value: float = 0.0,
        mode: str = "field",  # "field", "row", or "both"
        random_prob: float = 0.1,  # BERT-style: 10% random replacement
        keep_prob: float = 0.1,   # BERT-style: 10% keep original
    ):
        """
        Args:
            mask_prob: Overall masking probability (default: 0.15)
            field_mask_prob: Field-level masking probability (defaults to mask_prob)
            row_mask_prob: Row-level masking probability (default: 0.10)
            mask_token_id: Token ID for masked categorical features
            mask_value: Value for masked continuous features
            mode: Masking mode - "field", "row", or "both"
            random_prob: Probability of random replacement (BERT-style)
            keep_prob: Probability of keeping original token (BERT-style)
        """
        self.mask_prob = mask_prob
        self.field_mask_prob = field_mask_prob or mask_prob
        self.row_mask_prob = row_mask_prob
        self.mask_token_id = mask_token_id
        self.mask_value = mask_value
        self.mode = mode
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        
        # Ensure probabilities sum correctly for BERT-style masking
        self.mask_replace_prob = 1.0 - self.random_prob - self.keep_prob
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Apply masking to a batch of features.
        
        Args:
            features: List of feature dictionaries from dataset
            
        Returns:
            Dictionary with masked inputs, original targets, and mask information
        """
        # First, use the standard collate function to create the batch
        batch = collate_fn(features)
        
        # Extract tensors
        cat_input = batch["cat"]  # (B, L, C)
        cont_input = batch["cont"]  # (B, L, F)
        
        # For MLM, we need input and target sequences
        # Input: all tokens except the last one
        # Target: all tokens except the first one (shifted for next-token prediction)
        cat_input_seq = cat_input[:, :-1]  # (B, L-1, C)
        cont_input_seq = cont_input[:, :-1]  # (B, L-1, F)
        
        cat_target_seq = cat_input[:, 1:]  # (B, L-1, C) - targets for prediction
        cont_target_seq = cont_input[:, 1:]  # (B, L-1, F)
        
        # Handle quantization targets if present
        qtarget_seq = None
        if "qtarget" in batch:
            qtarget_seq = batch["qtarget"][:, 1:]  # (B, L-1, F)
        
        B, L, C = cat_input_seq.shape
        _, _, F = cont_input_seq.shape
        
        # Create masks based on the selected mode
        if self.mode == "field":
            mask = self._create_field_mask(B, L, C + F)
        elif self.mode == "row":
            mask = self._create_row_mask(B, L)
        elif self.mode == "both":
            mask = self._create_combined_mask(B, L, C + F)
        else:
            raise ValueError(f"Unknown masking mode: {self.mode}")
        
        # Apply BERT-style masking
        masked_cat_input, masked_cont_input = self._apply_bert_style_masking(
            cat_input_seq, cont_input_seq, mask, C
        )
        
        # Create labels (only compute loss on masked positions)
        cat_labels = cat_target_seq.clone()
        cont_labels = cont_target_seq.clone()
        qtarget_labels = qtarget_seq.clone() if qtarget_seq is not None else None
        
        # Set non-masked positions to -100 (ignored in loss computation)
        # Use torch.where to avoid in-place operations on tensors that require gradients
        if self.mode in ["field", "both"]:
            # Field-level mask: shape (B, L, C+F)
            cat_mask = mask[:, :, :C]  # (B, L, C)
            cont_mask = mask[:, :, C:]  # (B, L, F)
            
            # Use torch.where instead of in-place assignment
            cat_labels = torch.where(cat_mask, cat_labels, torch.full_like(cat_labels, -100))
            if qtarget_labels is not None:
                qtarget_labels = torch.where(cont_mask, qtarget_labels, torch.full_like(qtarget_labels, -100))
        else:
            # Row-level mask: shape (B, L)
            # Create separate expanded masks for different tensor shapes
            cat_mask_expanded = mask.unsqueeze(-1).expand_as(cat_labels)
            cat_labels = torch.where(cat_mask_expanded, cat_labels, torch.full_like(cat_labels, -100))
            if qtarget_labels is not None:
                qtarget_mask_expanded = mask.unsqueeze(-1).expand_as(qtarget_labels)
                qtarget_labels = torch.where(qtarget_mask_expanded, qtarget_labels, torch.full_like(qtarget_labels, -100))
        
        result = {
            "cat_input": masked_cat_input,
            "cont_input": masked_cont_input,
            "cat_labels": cat_labels,
            "cont_labels": cont_labels,
            "mask": mask,
            "label": batch["label"],  # Keep original labels for classification tasks
        }
        
        if qtarget_labels is not None:
            result["qtarget_labels"] = qtarget_labels
            
        return result
    
    def _create_field_mask(self, batch_size: int, seq_len: int, num_features: int) -> torch.Tensor:
        """Create field-level mask: randomly mask individual features."""
        mask = torch.rand(batch_size, seq_len, num_features) < self.field_mask_prob
        return mask
    
    def _create_row_mask(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Create row-level mask: randomly mask entire rows."""
        mask = torch.rand(batch_size, seq_len) < self.row_mask_prob
        return mask
    
    def _create_combined_mask(self, batch_size: int, seq_len: int, num_features: int) -> torch.Tensor:
        """Create combined field-level and row-level mask."""
        # Start with field-level masking
        mask = torch.zeros(batch_size, seq_len, num_features, dtype=torch.bool)
        
        # 1. Field-level masking: randomly mask individual fields
        field_mask = torch.rand(batch_size, seq_len, num_features) < self.field_mask_prob
        mask = mask | field_mask
        
        # 2. Row-level masking: randomly mask entire rows
        row_mask = torch.rand(batch_size, seq_len) < self.row_mask_prob  # (B, L)
        row_mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, num_features)  # (B, L, F)
        mask = mask | row_mask_expanded
        
        return mask
    
    def _apply_bert_style_masking(
        self,
        cat_input: torch.Tensor,
        cont_input: torch.Tensor,
        mask: torch.Tensor,
        num_cat_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BERT-style masking: 80% MASK, 10% random, 10% original.
        
        Args:
            cat_input: Categorical input tensor (B, L, C)
            cont_input: Continuous input tensor (B, L, F)
            mask: Boolean mask tensor
            num_cat_features: Number of categorical features
            
        Returns:
            Tuple of (masked_cat_input, masked_cont_input)
        """
        masked_cat_input = cat_input.clone()
        masked_cont_input = cont_input.clone()
        
        if self.mode in ["field", "both"]:
            # Field-level masking
            cat_mask = mask[:, :, :num_cat_features]  # (B, L, C)
            cont_mask = mask[:, :, num_cat_features:]  # (B, L, F)
            
            # Apply BERT-style masking to categorical features
            for i in range(num_cat_features):
                feature_mask = cat_mask[:, :, i]  # (B, L)
                masked_positions = feature_mask.nonzero(as_tuple=False)
                
                if len(masked_positions) > 0:
                    # Get the vocabulary size for this feature (approximate from data)
                    vocab_size = int(cat_input[:, :, i].max().item()) + 1
                    
                    # BERT-style masking probabilities
                    num_masked = len(masked_positions)
                    rand_vals = torch.rand(num_masked)
                    
                    # 80% mask with special token
                    mask_indices = rand_vals < self.mask_replace_prob
                    # 10% replace with random token
                    random_indices = (rand_vals >= self.mask_replace_prob) & (rand_vals < self.mask_replace_prob + self.random_prob)
                    # 10% keep original (do nothing)
                    
                    batch_idx = masked_positions[:, 0]
                    seq_idx = masked_positions[:, 1]
                    
                    # Apply mask tokens
                    masked_cat_input[batch_idx[mask_indices], seq_idx[mask_indices], i] = self.mask_token_id
                    
                    # Apply random tokens
                    if random_indices.any():
                        random_tokens = torch.randint(0, vocab_size, (int(random_indices.sum().item()),))
                        masked_cat_input[batch_idx[random_indices], seq_idx[random_indices], i] = random_tokens
            
            # Apply masking to continuous features (simpler - just mask with 0)
            for i in range(cont_input.shape[-1]):
                feature_mask = cont_mask[:, :, i]  # (B, L)
                masked_cont_input[:, :, i] = torch.where(
                    feature_mask,
                    torch.full_like(masked_cont_input[:, :, i], self.mask_value),
                    masked_cont_input[:, :, i]
                )
        
        else:
            # Row-level masking - mask shape is (B, L), broadcast to (B, L, 1) for each feature
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            
            # Apply mask to all categorical features
            for i in range(num_cat_features):
                masked_cat_input[:, :, i] = torch.where(
                    mask_expanded.squeeze(-1),
                    torch.full_like(masked_cat_input[:, :, i], self.mask_token_id),
                    masked_cat_input[:, :, i]
                )
            
            # Apply mask to all continuous features
            for i in range(cont_input.shape[-1]):
                masked_cont_input[:, :, i] = torch.where(
                    mask_expanded.squeeze(-1),
                    torch.full_like(masked_cont_input[:, :, i], self.mask_value),
                    masked_cont_input[:, :, i]
                )
        
        return masked_cat_input, masked_cont_input


def create_mlm_collator(
    mask_prob: float = 0.15,
    field_mask_prob: Optional[float] = None,
    row_mask_prob: float = 0.10,
    mode: str = "field"
) -> TabularDataCollatorForMLM:
    """
    Factory function to create an MLM data collator.
    
    Args:
        mask_prob: Overall masking probability
        field_mask_prob: Field-level masking probability (defaults to mask_prob)
        row_mask_prob: Row-level masking probability
        mode: Masking mode - "field", "row", or "both"
        
    Returns:
        TabularDataCollatorForMLM instance
    """
    return TabularDataCollatorForMLM(
        mask_prob=mask_prob,
        field_mask_prob=field_mask_prob,
        row_mask_prob=row_mask_prob,
        mode=mode
    ) 