#!/usr/bin/env python
"""
Debug script to visualize exactly what the field-level and row-level masking is doing.
"""
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.masking import create_field_and_row_mask

def debug_masking():
    """Debug the masking to see exactly what's happening."""
    
    # Small test case
    batch_size = 1
    seq_len = 10  # 10 transactions
    num_features = 14  # 13 categorical + 1 continuous
    
    cat_features = ["User", "Card", "Use Chip", "Merchant Name", "Merchant City", 
                   "Merchant State", "Zip", "MCC", "Errors?", "Year", "Month", "Day", "Hour"]
    cont_features = ["Amount"]
    
    print(f"Testing masking with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Total features: {num_features}")
    print(f"  Categorical features: {len(cat_features)}")
    print(f"  Continuous features: {len(cont_features)}")
    print()
    
    # Test different masking probabilities
    test_cases = [
        (0.15, 0.10),  # Default: 15% field, 10% row
        (0.10, 0.05),  # Lower: 10% field, 5% row
        (0.05, 0.02),  # Very low: 5% field, 2% row
    ]
    
    for field_prob, row_prob in test_cases:
        print(f"=" * 80)
        print(f"TESTING: Field masking {field_prob*100:.0f}%, Row masking {row_prob*100:.0f}%")
        print(f"=" * 80)
        
        # Create mask
        mask = create_field_and_row_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            field_mask_prob=field_prob,
            row_mask_prob=row_prob,
            device=torch.device("cpu")
        )
        
        print(f"Mask shape: {mask.shape}")  # Should be (1, 10, 14)
        
        # Analyze the mask
        sample_mask = mask[0]  # (seq_len, num_features) = (10, 14)
        
        # Check which rows are completely masked (all features masked)
        rows_fully_masked = []
        for seq_idx in range(seq_len):
            if sample_mask[seq_idx].all():
                rows_fully_masked.append(seq_idx)
        
        print(f"Fully masked rows (entire transactions): {rows_fully_masked}")
        
        # Check which features are masked at least once
        features_masked = []
        for feat_idx in range(num_features):
            if sample_mask[:, feat_idx].any():
                feat_name = cat_features[feat_idx] if feat_idx < len(cat_features) else cont_features[feat_idx - len(cat_features)]
                features_masked.append(feat_name)
        
        print(f"Features masked at least once: {len(features_masked)}")
        print(f"Masked features: {features_masked}")
        
        # Show detailed mask visualization
        print("\nDetailed mask visualization:")
        print("Rows (transactions) →")
        print("Features ↓")
        
        # Create header
        header = "Feature".ljust(15)
        for seq_idx in range(seq_len):
            header += f"T{seq_idx}".ljust(3)
        print(header)
        
        # Show each feature's mask across all transactions
        for feat_idx in range(num_features):
            feat_name = cat_features[feat_idx] if feat_idx < len(cat_features) else cont_features[feat_idx - len(cat_features)]
            row = feat_name[:14].ljust(15)
            
            for seq_idx in range(seq_len):
                is_masked = sample_mask[seq_idx, feat_idx].item()
                row += ("X" if is_masked else ".").ljust(3)
            
            print(row)
        
        # Statistics
        total_positions = seq_len * num_features
        masked_positions = sample_mask.sum().item()
        mask_percentage = (masked_positions / total_positions) * 100
        
        print(f"\nStatistics:")
        print(f"  Total positions: {total_positions}")
        print(f"  Masked positions: {masked_positions}")
        print(f"  Actual mask percentage: {mask_percentage:.1f}%")
        print(f"  Expected field mask percentage: {field_prob*100:.0f}%")
        print(f"  Expected row mask percentage: {row_prob*100:.0f}%")
        
        # Calculate expected mask percentage
        # P(field OR row) = P(field) + P(row) - P(field AND row)
        expected_percentage = (field_prob + row_prob - field_prob * row_prob) * 100
        print(f"  Expected combined percentage: {expected_percentage:.1f}%")
        print()


if __name__ == "__main__":
    debug_masking() 