#!/usr/bin/env python3
"""
Data Leakage Inspection Script

This script loads the saved train_df, val_df, test_df for both legit and full datasets
and performs comprehensive data leakage analysis.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

def load_processed_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Load processed data from .pt file."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    train_df, val_df, test_df, schema = data
    return train_df, val_df, test_df, schema

def basic_info(df: pd.DataFrame, name: str) -> None:
    """Print basic information about a dataframe."""
    print(f"\n{'='*50}")
    print(f"BASIC INFO: {name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Columns: {list(df.columns)}")
    print(f"Index type: {type(df.index)}")
    print(f"Index sample: {df.index[:5].tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values:\n{missing[missing > 0]}")
    else:
        print("No missing values found")

def check_data_leakage(full_dfs: Dict[str, pd.DataFrame], 
                      legit_dfs: Dict[str, pd.DataFrame]) -> None:
    """Check for data leakage between datasets using content-based comparison."""
    print(f"\n{'='*50}")
    print("DATA LEAKAGE ANALYSIS")
    print(f"{'='*50}")
    
    for split in ['train', 'val', 'test']:
        print(f"\n--- {split.upper()} SPLIT ---")
        
        full_df = full_dfs[split]
        legit_df = legit_dfs[split]
        
        print(f"Full {split} size: {len(full_df)}")
        print(f"Legit {split} size: {len(legit_df)}")
        
        # Check if legit is subset of full (content-based)
        if split == 'train':
            print(f"Checking if legit train is subset of full train...")
            
            # Sample-based comparison for performance
            sample_size = min(100, len(legit_df))
            legit_sample = legit_df.head(sample_size)
            full_sample = full_df.head(1000)  # Larger sample from full
            
            legit_in_full_count = 0
            for _, legit_row in legit_sample.iterrows():
                for _, full_row in full_sample.iterrows():
                    if legit_row.equals(full_row):
                        legit_in_full_count += 1
                        break
            
            print(f"Legit rows found in full: {legit_in_full_count}/{len(legit_sample)}")
            
            if legit_in_full_count == len(legit_sample):
                print("OK: All sampled legit rows found in full dataset")
            elif legit_in_full_count > 0:
                print("PARTIAL: Some legit rows found in full dataset")
            else:
                print("WARNING: No legit rows found in full dataset")
        
        # Check for overlap between splits (content-based)
        if split == 'train':
            print(f"\nChecking for duplicate rows between splits...")
            
            # Sample-based comparison
            sample_size = min(500, len(full_df), len(full_dfs['val']), len(full_dfs['test']))
            
            train_sample = full_df.head(sample_size)
            val_sample = full_dfs['val'].head(sample_size)
            test_sample = full_dfs['test'].head(sample_size)
            
            # Check for duplicate rows between splits
            train_val_duplicates = 0
            train_test_duplicates = 0
            val_test_duplicates = 0
            
            # Train vs Val
            for _, train_row in train_sample.iterrows():
                for _, val_row in val_sample.iterrows():
                    if train_row.equals(val_row):
                        train_val_duplicates += 1
                        break
            
            # Train vs Test
            for _, train_row in train_sample.iterrows():
                for _, test_row in test_sample.iterrows():
                    if train_row.equals(test_row):
                        train_test_duplicates += 1
                        break
            
            # Val vs Test
            for _, val_row in val_sample.iterrows():
                for _, test_row in test_sample.iterrows():
                    if val_row.equals(test_row):
                        val_test_duplicates += 1
                        break
            
            print(f"Duplicate rows found (sample of {sample_size}):")
            print(f"  Train-Val: {train_val_duplicates}")
            print(f"  Train-Test: {train_test_duplicates}")
            print(f"  Val-Test: {val_test_duplicates}")
            
            if train_val_duplicates > 0 or train_test_duplicates > 0 or val_test_duplicates > 0:
                print("WARNING: Duplicate rows detected between splits!")
            else:
                print("OK: No duplicate rows found in sample")

def analyze_feature_distributions(full_dfs: Dict[str, pd.DataFrame], 
                                legit_dfs: Dict[str, pd.DataFrame]) -> None:
    """Analyze feature distributions for potential leakage."""
    print(f"\n{'='*50}")
    print("FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*50}")
    
    # Get numeric columns
    sample_df = full_dfs['train']
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Analyzing {len(numeric_cols)} numeric features...")
    
    for col in numeric_cols[:10]:  # Limit to first 10 for readability
        print(f"\n--- {col} ---")
        
        full_train_vals = full_dfs['train'][col].dropna()
        legit_train_vals = legit_dfs['train'][col].dropna()
        
        print(f"Full train: mean={full_train_vals.mean():.4f}, std={full_train_vals.std():.4f}")
        print(f"Legit train: mean={legit_train_vals.mean():.4f}, std={legit_train_vals.std():.4f}")
        
        # Check if distributions are similar
        if abs(full_train_vals.mean() - legit_train_vals.mean()) > 0.1:
            print("WARNING: Significant mean difference detected")

def check_schema_consistency(full_schema: Any, legit_schema: Any) -> None:
    """Check if schemas are consistent."""
    print(f"\n{'='*50}")
    print("SCHEMA CONSISTENCY CHECK")
    print(f"{'='*50}")
    
    print(f"Full schema type: {type(full_schema)}")
    print(f"Legit schema type: {type(legit_schema)}")
    
    # Add more schema-specific checks here based on your schema structure
    if hasattr(full_schema, 'cat_features'):
        print(f"Full cat features: {len(full_schema.cat_features)}")
        print(f"Legit cat features: {len(legit_schema.cat_features)}")
    
    if hasattr(full_schema, 'cont_features'):
        print(f"Full cont features: {len(full_schema.cont_features)}")
        print(f"Legit cont features: {len(legit_schema.cont_features)}")

def main():
    """Main function to run data leakage inspection."""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found!")
        sys.exit(1)
    
    full_path = data_dir / "full_processed.pt"
    legit_path = data_dir / "legit_processed.pt"
    
    if not full_path.exists():
        print(f"Error: {full_path} not found!")
        sys.exit(1)
    
    if not legit_path.exists():
        print(f"Error: {legit_path} not found!")
        sys.exit(1)
    
    # Load data
    print("Loading processed datasets...")
    full_train, full_val, full_test, full_schema = load_processed_data(str(full_path))
    legit_train, legit_val, legit_test, legit_schema = load_processed_data(str(legit_path))
    
    # Organize dataframes
    full_dfs = {'train': full_train, 'val': full_val, 'test': full_test}
    legit_dfs = {'train': legit_train, 'val': legit_val, 'test': legit_test}
    
    # Basic information
    for name, df in full_dfs.items():
        basic_info(df, f"Full {name}")
    
    for name, df in legit_dfs.items():
        basic_info(df, f"Legit {name}")
    
    # Data leakage checks
    check_data_leakage(full_dfs, legit_dfs)
    
    # Feature distribution analysis
    analyze_feature_distributions(full_dfs, legit_dfs)
    
    # Schema consistency
    check_schema_consistency(full_schema, legit_schema)
    
    print(f"\n{'='*50}")
    print("INSPECTION COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 