#!/usr/bin/env python3
"""
Comprehensive Data Leakage Check (Row-level & User-level)

This script creates TxnDataset objects for all 6 datasets (full train/val/test and legit train/val/test)
and performs extensive comparison between them to detect data leakage.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import time
import hashlib

# Import project modules
sys.path.append('src')
from transaction_transformer.data.dataset import TxnDataset
from transaction_transformer.data.preprocessing import FieldSchema

def load_processed_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Load processed data from .pt file."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    train_df, val_df, test_df, schema = data
    return train_df, val_df, test_df, schema

def create_txn_datasets(dfs: Dict[str, pd.DataFrame], schema: FieldSchema, 
                       dataset_name: str) -> Dict[str, TxnDataset]:
    """Create TxnDataset objects for train/val/test splits."""
    print(f"\nCreating TxnDataset objects for {dataset_name}...")
    
    datasets = {}
    for split_name, df in dfs.items():
        print(f"  Creating {split_name} dataset...")
        try:
            # Assuming 'user_id' is the group_by column - adjust if different
            dataset = TxnDataset(
                df=df,
                                 group_by="User",
                 schema=schema,
                 window=10,
                 stride=5,
                 include_all_fraud=True
            )
            datasets[split_name] = dataset
            print(f"    Created {split_name} dataset with {len(dataset)} samples")
        except Exception as e:
            print(f"    Error creating {split_name} dataset: {e}")
            datasets[split_name] = None
    
    return datasets

def compute_dataset_hashes(dataset: TxnDataset) -> Set[str]:
    """Compute a hash for every sample in the dataset and return a set of hashes."""
    if dataset is None:
        return set()
    
    hash_set: Set[str] = set()
    n = len(dataset)
    print(f"    Computing hashes for {n} windows…")
    for idx in range(n):
        sample = dataset[idx]
        # Convert tensors to bytes
        cat_bytes = sample['cat'].contiguous().numpy().tobytes()
        cont_bytes = sample['cont'].contiguous().numpy().tobytes()
        label_bytes = sample['label'].contiguous().numpy().tobytes() if hasattr(sample['label'], 'numpy') else int(sample['label']).to_bytes(2, 'little')
        m = hashlib.blake2b(digest_size=16)
        m.update(cat_bytes)
        m.update(cont_bytes)
        m.update(label_bytes)
        hash_set.add(m.hexdigest())
    return hash_set

# -----------------------------------------------------------------------------
# Legacy extraction function kept for debugging (not used in main flow)
# -----------------------------------------------------------------------------
def extract_window_data(dataset: TxnDataset, max_samples: int = 1000) -> List[Dict]:
    """Extract window data from dataset for comparison."""
    if dataset is None:
        return []
    
    samples = []
    sample_indices = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=int)
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            samples.append({
                'cat': sample['cat'].numpy(),
                'cont': sample['cont'].numpy(),
                'label': sample['label'].item()
            })
        except Exception as e:
            print(f"Error extracting sample {idx}: {e}")
            continue
    
    return samples

def compare_window_data(samples1: List[Dict], samples2: List[Dict], 
                       name1: str, name2: str) -> Dict:
    """Compare two sets of window data for duplicates."""
    print(f"  Comparing {name1} vs {name2} ({len(samples1)} vs {len(samples2)} samples)...")
    
    duplicates = []
    duplicate_count = 0
    
    for i, sample1 in enumerate(samples1):
        for j, sample2 in enumerate(samples2):
            # Compare categorical data
            if np.array_equal(sample1['cat'], sample2['cat']):
                # Compare continuous data (with tolerance for floating point)
                if np.allclose(sample1['cont'], sample2['cont'], rtol=1e-5, atol=1e-8):
                    # Compare labels
                    if sample1['label'] == sample2['label']:
                        duplicates.append({
                            'sample1_idx': i,
                            'sample2_idx': j,
                            'cat_shape': sample1['cat'].shape,
                            'cont_shape': sample1['cont'].shape,
                            'label': sample1['label']
                        })
                        duplicate_count += 1
                        break  # Found duplicate, move to next sample1
    
    return {
        'duplicate_count': duplicate_count,
        'duplicates': duplicates,
        'sample1_total': len(samples1),
        'sample2_total': len(samples2)
    }




def check_chronological_order(df: pd.DataFrame, name: str) -> None:
    """Ensure each user's rows are chronologically sorted."""
    print(f"  Chronological order check for {name} …")
    bad_users = []
    grouped = df.groupby('User', sort=False)
    for user, grp in grouped:
        # create tuple timestamp order list
        times = list(zip(grp['Year'], grp['Month'], grp['Day'], grp['Hour']))
        if times != sorted(times):
            bad_users.append(user)
            if len(bad_users) >= 5:
                break
    if bad_users:
        print(f"    WARNING: {len(bad_users)} users have out-of-order timestamps (sample: {bad_users[:5]})")
    else:
        print("    OK: All user sequences sorted chronologically")


def compute_row_hashes(df: pd.DataFrame) -> Set[str]:
    """Hash each row content to detect duplicates across splits."""
    hashes: Set[str] = set()
    m = hashlib.blake2b
    arr = df.to_numpy()
    for row in arr:
        h = m(row.tobytes(), digest_size=16).hexdigest()
        hashes.add(h)
    return hashes


def row_level_checks(dfs: Dict[str, pd.DataFrame], dataset_name: str) -> None:
    """Check for duplicate rows across splits within a dataset type."""
    print(f"\nRow-duplication check for {dataset_name} dataset …")
    row_hashes = {split: compute_row_hashes(df) for split, df in dfs.items()}
    pairs = [('train','val'), ('train','test'), ('val','test')]
    leak = False
    for a,b in pairs:
        dup = row_hashes[a] & row_hashes[b]
        if dup:
            print(f"  WARNING: {len(dup)} duplicate rows between {a} and {b} splits")
            leak = True
    if not leak:
        print("  OK: No duplicate rows across splits")


# -----------------------------------------------------------------------------
# Main exhaustive comparison (hash-based windows) retained for window leakage
# -----------------------------------------------------------------------------

def compare_all_datasets(full_datasets: Dict[str, TxnDataset], 
                         legit_datasets: Dict[str, TxnDataset]) -> None:
    """Perform comprehensive comparison between all datasets."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE DATA LEAKAGE ANALYSIS")
    print(f"{'='*80}")
    
    # Compute hashes for all datasets
    print("\nComputing hashes for all datasets (this may take a while)…")
    hash_sets: Dict[str, Set[str]] = {}
    
    for split in ['train', 'val', 'test']:
        if full_datasets[split] is not None:
            name = f'full_{split}'
            hash_sets[name] = compute_dataset_hashes(full_datasets[split])
            print(f"  {name}: {len(hash_sets[name])} unique hashes")
        if legit_datasets[split] is not None:
            name = f'legit_{split}'
            hash_sets[name] = compute_dataset_hashes(legit_datasets[split])
            print(f"  {name}: {len(hash_sets[name])} unique hashes")
    
    # Compare all pairs of hash sets
    print(f"\nComparing all dataset pairs (hash‐based)…")
    comparison_results = {}
    dataset_names = list(hash_sets.keys())
    for i, name1 in enumerate(dataset_names):
        for j, name2 in enumerate(dataset_names):
            if i < j:
                comparison_key = f"{name1}_vs_{name2}"
                set1 = hash_sets[name1]
                set2 = hash_sets[name2]
                
                intersect = set1.intersection(set2)
                dup_count = len(intersect)
                comp_result = {
                    'duplicate_count': dup_count,
                    'set1_size': len(set1),
                    'set2_size': len(set2)
                }
                comparison_results[comparison_key] = comp_result
                
                print(f"  {comparison_key}: {dup_count} duplicates (|A|={len(set1)}, |B|={len(set2)})")
    
    # Summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    total_comparisons = len(comparison_results)
    comparisons_with_duplicates = sum(1 for r in comparison_results.values() if r['duplicate_count'] > 0)
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Comparisons with duplicates: {comparisons_with_duplicates}")
    
    if comparisons_with_duplicates > 0:
        print(f"\nWARNING: Data leakage detected in {comparisons_with_duplicates} comparisons!")
        for comparison_key, result in comparison_results.items():
            if result['duplicate_count'] > 0:
                print(f"  {comparison_key}: {result['duplicate_count']} duplicates (|A|={result['set1_size']}, |B|={result['set2_size']})")
    else:
        print(f"\nOK: No data leakage detected across all comparisons!")

def analyze_dataset_statistics(full_datasets: Dict[str, TxnDataset], 
                             legit_datasets: Dict[str, TxnDataset]) -> None:
    """Analyze basic statistics of all datasets."""
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    
    for dataset_name, datasets in [("Full", full_datasets), ("Legit", legit_datasets)]:
        print(f"\n{dataset_name} Datasets:")
        for split, dataset in datasets.items():
            if dataset is not None:
                print(f"  {split}: {len(dataset)} samples")
                
                # Sample a few items to get shapes
                try:
                    sample = dataset[0]
                    print(f"    Cat shape: {sample['cat'].shape}")
                    print(f"    Cont shape: {sample['cont'].shape}")
                    print(f"    Label: {sample['label']}")
                except Exception as e:
                    print(f"    Error sampling: {e}")
            else:
                print(f"  {split}: Failed to create dataset")

def main():
    """Main function to run comprehensive data leakage check."""
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
    
    # ----------------------------------
    # Row-level leakage checks first (DataFrame based)
    # ----------------------------------
    # User overlap is allowed; we only forbid exact transaction duplication across splits.

    # Chronological order checks
    for split, df in full_dfs.items():
        check_chronological_order(df, f"Full {split}")
    for split, df in legit_dfs.items():
        check_chronological_order(df, f"Legit {split}")

    # Row duplication across splits
    row_level_checks(full_dfs, "Full")
    row_level_checks(legit_dfs, "Legit")

    # Create TxnDataset objects
    print(f"\nCreating TxnDataset objects...")
    full_datasets = create_txn_datasets(full_dfs, full_schema, "Full")
    legit_datasets = create_txn_datasets(legit_dfs, legit_schema, "Legit")
    
    # Analyze dataset statistics
    analyze_dataset_statistics(full_datasets, legit_datasets)
    
    # Perform comprehensive comparison
    compare_all_datasets(full_datasets, legit_datasets)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 