#!/usr/bin/env python3
"""
Test script to verify data loading works correctly with the updated paths.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pretrain_data_loading():
    """Test loading of legitimate transactions data for pretraining."""
    print("Testing pretrain data loading...")
    
    data_path = project_root / "data" / "datasets" / "legitimate_transactions_processed.pt"
    
    if not data_path.exists():
        print(f"Pretrain data not found at: {data_path}")
        print("   Please run preprocessing.py first to create the data files.")
        return False
    
    try:
        data = torch.load(data_path, weights_only=False)
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams = data
        
        print(f"Pretrain data loaded successfully!")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Val samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Categorical features: {len(cat_features)}")
        print(f"   Continuous features: {len(cont_features)}")
        print(f"   Encoders: {list(enc.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error loading pretrain data: {e}")
        return False

def test_finetune_data_loading():
    """Test loading of all transactions data for finetuning."""
    print("\nTesting finetune data loading...")
    
    data_path = project_root / "data" / "datasets" / "all_transactions_processed.pt"
    
    if not data_path.exists():
        print(f"Finetune data not found at: {data_path}")
        print("   Please run preprocessing.py first to create the data files.")
        return False
    
    try:
        data = torch.load(data_path, weights_only=False)
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler, qparams = data
        
        print(f"Finetune data loaded successfully!")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Val samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Categorical features: {len(cat_features)}")
        print(f"   Continuous features: {len(cont_features)}")
        print(f"   Encoders: {list(enc.keys())}")
        
        # Check if fraud labels exist
        if 'label' in train_df.columns:
            fraud_ratio = train_df['label'].mean()
            print(f"   Fraud ratio in training: {fraud_ratio:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error loading finetune data: {e}")
        return False

def main():
    """Run all data loading tests."""
    print("Testing Data Loading...")
    print("=" * 50)
    
    pretrain_ok = test_pretrain_data_loading()
    finetune_ok = test_finetune_data_loading()
    
    print("\n" + "=" * 50)
    if pretrain_ok and finetune_ok:
        print("All data loading tests passed!")
        print("\nYou can now run training:")
        print("  python training/pretrain.py --config configs/pretrain.yaml --mode ar")
        print("  python training/pretrain.py --config configs/pretrain.yaml --mode masked")
        print("  python training/finetune.py --config configs/finetune.yaml")
    else:
        print("Some data loading tests failed.")
        print("Please run preprocessing.py first to create the required data files.")

if __name__ == "__main__":
    main() 