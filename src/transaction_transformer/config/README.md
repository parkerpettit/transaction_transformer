# Configuration System

This directory contains the configuration system for the Transaction Transformer project. The system provides a clean, type-safe way to manage configuration with support for YAML files and CLI overrides.

## Overview

The configuration system consists of:

- **Config Classes**: Type-safe dataclasses defining the configuration structure
- **ConfigManager**: Handles loading, merging, and validation of configurations
- **CLI Integration**: Seamless integration with command-line arguments
- **YAML Support**: Configuration files in YAML format

## Quick Start

### Basic Usage

```python
from transaction_transformer.config.cli import get_config

# Load configuration (uses pretrain.yaml by default)
config = get_config()

# Access configuration values
print(f"Model type: {config.model.model_type}")
print(f"Batch size: {config.model.training.batch_size}")
print(f"Learning rate: {config.model.training.learning_rate}")
```

### With CLI Overrides

```bash
# Override specific parameters
python your_script.py --batch-size 64 --learning-rate 0.001 --training-mode mlm

# Use different config file
python your_script.py --config finetune.yaml --device cuda
```

### Custom Config File

```python
# Load from specific config file
config = get_config("finetune.yaml")
```

## Configuration Structure

The configuration is organized into logical sections:

### Model Configuration (`config.model`)

- **model_type**: Type of model ("feature_prediction", "fraud_detection")
- **field_transformer**: Field transformer parameters
- **sequence_transformer**: Sequence transformer parameters
- **embedding**: Embedding layer configuration
- **training**: Training parameters
- **data**: Data loading parameters

### Metrics Configuration (`config.metrics`)

- **experiment_name**: Name for experiment tracking
- **wandb**: Weights & Biases integration settings
- **tensorboard**: TensorBoard logging settings
- **seed**: Random seed for reproducibility

## Configuration Files

### pretrain.yaml
Default configuration for pretraining models. Includes:
- AR/MLM training modes
- Transformer architecture settings
- Data preprocessing parameters

### finetune.yaml
Configuration for fine-tuning on downstream tasks. Includes:
- Fraud detection settings
- Lower learning rates
- Different model architecture

## CLI Arguments

The system supports CLI arguments that override YAML settings:

### Training Parameters
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--total-epochs`: Number of training epochs
- `--training-mode`: "mlm" or "ar"
- `--device`: "cpu", "cuda", or "auto"

### Model Architecture
- `--field-d-model`: Field transformer dimension
- `--field-n-heads`: Field transformer heads
- `--field-depth`: Field transformer depth
- `--seq-d-model`: Sequence transformer dimension
- `--seq-n-heads`: Sequence transformer heads
- `--seq-depth`: Sequence transformer depth

### Data Parameters
- `--data-dir`: Data directory
- `--window`: Sequence window size
- `--stride`: Stride between windows
- `--num-bins`: Number of bins for continuous features

### MLM Parameters
- `--p-field`: Field masking probability
- `--p-row`: Row masking probability

### Experiment Tracking
- `--experiment-name`: Experiment name
- `--run-name`: Run name
- `--wandb`: Enable wandb logging
- `--tensorboard`: Enable tensorboard logging
- `--seed`: Random seed

## Advanced Usage

### Using ConfigManager

For more control over configuration loading:

```python
from transaction_transformer.config.cli import get_config_manager

config_manager = get_config_manager("custom_config.yaml")
config = config_manager.load_config()

# Access raw config dict
config_dict = config.to_dict()
```

### Configuration Validation

The system automatically validates configuration:

```python
# This will raise ValueError if config is invalid
config = get_config()

# Manual validation
config._validate_config()
```

### Device Detection

```python
# Automatic device detection
device = config.get_device()  # Returns "cuda" or "cpu"
```

## Best Practices

1. **Use YAML for defaults**: Keep all default values in YAML files
2. **CLI for overrides**: Use CLI arguments for quick experiments
3. **Type safety**: Always use the Config object, not raw dictionaries
4. **Validation**: Let the system validate your configuration
5. **Consistency**: Use consistent naming across config files

## Migration from Old System

If you're migrating from the old configuration system:

1. Replace `get_config()` calls with the new system
2. Remove manual type conversions (handled automatically)
3. Use `config.to_dict()` for wandb logging
4. Use `config.get_device()` for device detection

## Example Integration

```python
# In your training script
from transaction_transformer.config.cli import get_config

def main():
    # Load configuration
    config = get_config()
    
    # Initialize wandb
    if config.metrics.wandb:
        wandb.init(
            project=config.metrics.wandb_project,
            config=config.to_dict()
        )
    
    # Create model
    model = YourModel(config.model)
    
    # Train
    trainer = YourTrainer(
        model=model,
        config=config.model.training,
        device=config.get_device()
    )
    
    trainer.train(config.model.training.total_epochs)
```

## Troubleshooting

### Common Issues

1. **Type errors**: Ensure you're using the Config object, not raw dict
2. **Missing values**: Check that your YAML file has all required fields
3. **Validation errors**: Check probability ranges and valid choices
4. **CLI not working**: Ensure argument names match the mapping in ConfigManager

### Debug Configuration

```python
# Print full configuration
config = get_config()
print(config.to_dict())

# Check specific values
print(f"Training mode: {config.model.training.mode}")
print(f"Device: {config.get_device()}")
``` 