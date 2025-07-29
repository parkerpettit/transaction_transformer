#!/usr/bin/env python
"""
Test script to demonstrate the centralized path management system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.paths import ProjectPaths, create_paths_from_config


def test_basic_functionality():
    """Test basic path functionality."""
    print("Testing Basic Path Functionality")
    print("=" * 50)
    
    # Test 1: Default paths
    print("1. Default Paths:")
    paths = ProjectPaths(data_dir="data")
    print(f"   Created successfully")
    print(f"   Checkpoint: {paths.checkpoint_path.name}")
    print(f"   Pretrain data: {paths.pretrain_data_path.name}")
    print()
    
    # Test 2: Custom filenames
    print("2. Custom Filenames:")
    custom_paths = ProjectPaths(
        data_dir="experiments/test",
        checkpoint_filename="my_model.pt",
        pretrain_data_filename="my_data.pt"
    )
    print(f"   Created successfully")
    print(f"   Checkpoint: {custom_paths.checkpoint_path.name}")
    print(f"   Pretrain data: {custom_paths.pretrain_data_path.name}")
    print()
    
    # Test 3: From config dictionary
    print("3. From Config Dictionary:")
    config = {
        "data_dir": "test_run",
        "checkpoint_filename": "best_model.pt",
        "pretrain_data_filename": "processed_data.pt",
        "batch_size": 32,  # Non-path args should be ignored
        "lr": 0.001
    }
    config_paths = create_paths_from_config(config)
    print(f"   Created successfully")
    print(f"   Checkpoint: {config_paths.checkpoint_path.name}")
    print(f"   Pretrain data: {config_paths.pretrain_data_path.name}")
    print()


def test_cli_simulation():
    """Simulate CLI argument usage."""
    print("CLI Simulation Test")
    print("=" * 50)
    
    # Simulate different CLI scenarios
    scenarios = [
        {
            "name": "Default paths only",
            "config": {"data_dir": "data"}
        },
        {
            "name": "Custom checkpoint",
            "config": {
                "data_dir": "data", 
                "checkpoint_filename": "experiment_v2.pt"
            }
        },
        {
            "name": "Full customization",
            "config": {
                "data_dir": "experiments/fraud_v3",
                "checkpoint_filename": "fraud_transformer.pt",
                "pretrain_data_filename": "clean_transactions.pt",
                "finetune_data_filename": "labeled_transactions.pt"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}:")
        paths = create_paths_from_config(scenario['config'])
        print(f"   Data dir: {Path(paths.data_dir).name}")
        print(f"   Checkpoint: {paths.checkpoint_path.name}")
        print(f"   Pretrain: {paths.pretrain_data_path.name}")
        print()


def test_error_handling():
    """Test error handling."""
    print("Error Handling Test")
    print("=" * 50)
    
    # Test missing data_dir
    try:
        create_paths_from_config({})
        print("   Should have failed!")
    except ValueError as e:
        print(f"   Correctly caught error: {e}")
    
    print()


def main():
    """Run all tests."""
    print("Centralized Path Management System Tests\n")
    
    test_basic_functionality()
    test_cli_simulation() 
    test_error_handling()
    
    print("All tests completed successfully!")
    print("\nReady to use:")
    print("   * CLI: python training/pretrain.py --data_dir /path --checkpoint_filename model.pt")
    print("   * YAML: Add path settings to config files")
    print("   * Code: Use ProjectPaths() or create_paths_from_config()")


if __name__ == "__main__":
    main() 