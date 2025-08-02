"""
Example usage of the new configuration system.

This shows how to:
1. Load configuration from YAML file
2. Override with CLI arguments
3. Use the Config object in your code
"""

from transaction_transformer.config.cli import get_config, get_config_manager
from transaction_transformer.config.config import Config


def example_basic_usage():
    """Basic usage example."""
    # Load config with default file (pretrain.yaml)
    config = get_config()
    
    # Access configuration values
    print(f"Model type: {config.model.model_type}")
    print(f"Training mode: {config.model.training.mode}")
    print(f"Batch size: {config.model.training.batch_size}")
    print(f"Learning rate: {config.model.training.learning_rate}")
    print(f"Device: {config.get_device()}")
    
    # Access nested configuration
    print(f"Field transformer d_model: {config.model.field_transformer.d_model}")
    print(f"Sequence transformer depth: {config.model.sequence_transformer.depth}")


def example_custom_config_file():
    """Example with custom config file."""
    # Load from specific config file
    config = get_config("finetune.yaml")
    
    print(f"Using finetune config: {config.model.model_type}")
    print(f"Training mode: {config.model.training.mode}")


def example_config_manager():
    """Example using ConfigManager for more control."""
    config_manager = get_config_manager("pretrain.yaml")
    
    # You can access the raw config dict if needed
    config_dict = config_manager.config.to_dict() if config_manager.config else {}
    print(f"Config has {len(config_dict)} top-level keys")
    
    # Load the config
    config = config_manager.load_config()
    print(f"Loaded config for: {config.model.model_type}")


def example_cli_overrides():
    """
    Example showing how CLI arguments override config file.
    
    Run this with: python -m transaction_transformer.config.example_usage --batch-size 64 --learning-rate 0.001
    """
    config = get_config()
    
    print("Configuration loaded with CLI overrides:")
    print(f"Batch size: {config.model.training.batch_size}")
    print(f"Learning rate: {config.model.training.learning_rate}")
    print(f"Training mode: {config.model.training.mode}")


def example_validation():
    """Example showing configuration validation."""
    try:
        # This would fail validation if the config had invalid values
        config = get_config()
        print("Configuration is valid!")
        
        # Test device detection
        device = config.get_device()
        print(f"Using device: {device}")
        
    except ValueError as e:
        print(f"Configuration validation failed: {e}")


if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()
    
    print("\n=== Custom Config File ===")
    example_custom_config_file()
    
    print("\n=== Config Manager ===")
    example_config_manager()
    
    print("\n=== CLI Overrides ===")
    example_cli_overrides()
    
    print("\n=== Validation ===")
    example_validation() 