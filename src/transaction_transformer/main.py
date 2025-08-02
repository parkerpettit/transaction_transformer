"""
Main entry point for transaction transformer.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from transaction_transformer.config import parse_cli_args, ConfigManager
from transaction_transformer.modeling.training import pretrain, finetune


def main():
    """Main entry point."""
    # Parse CLI arguments
    cli_args = parse_cli_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(cli_args["config"])
    
    # Merge CLI args with config
    config = config_manager.merge_configs(config, cli_args)
    
    # Validate configuration
    config_manager.validate_config(config)
    
    # Run appropriate training mode
    mode = cli_args["mode"]
    if mode == "pretrain":
        pretrain(config)
    elif mode == "finetune":
        finetune(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main() 