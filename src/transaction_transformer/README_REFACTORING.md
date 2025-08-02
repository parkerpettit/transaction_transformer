# Transaction Transformer - Refactored Structure

## Overview

This document outlines the new modular structure for the transaction transformer codebase. The refactoring focuses on:

1. **Modularity**: Each component is isolated and can be tested independently
2. **Extensibility**: Easy to add new model types, training modes, or components
3. **Maintainability**: Clear separation of concerns and smaller, focused files
4. **Configuration**: CLI-driven with config file fallbacks
5. **Training Modes**: Support for pretraining, finetuning, autoregressive, and MLM training

## New File Structure

```
src/transaction_transformer/
├── __init__.py
├── main.py                          # Main entry point
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── base.py                      # Base configuration classes
│   ├── cli.py                       # CLI argument parsing
│   └── config_manager.py            # Configuration loading/validation
├── data/                            # Data pipeline
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base_dataset.py          # Abstract base dataset
│   │   ├── transaction_dataset.py   # Transaction-specific dataset
│   │   └── fraud_dataset.py        # Fraud-specific dataset
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── batch_sampler.py        # Custom batch samplers
│   └── preprocessing/
│       ├── __init__.py
│       ├── schema.py               # FieldSchema and related
│       ├── tokenizer.py            # Tokenization logic
│       └── transforms.py           # Data transformations
├── modeling/                        # Model and training
│   ├── __init__.py
│   ├── models/                      # Model components (already split)
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py
│   │   │   ├── field_transformer.py
│   │   │   ├── sequence_transformer.py
│   │   │   ├── projection.py
│   │   │   └── heads.py
│   │   ├── transaction_embedding.py
│   │   ├── transaction_prediction.py
│   │   └── fraud_detection.py
│   └── training/
│       ├── __init__.py
│       ├── base/
│       │   ├── __init__.py
│       │   ├── base_trainer.py     # Abstract base trainer
│       │   ├── checkpoint_manager.py # Save/load checkpoint logic
│       │   └── metrics.py          # Training metrics and logging
│       ├── trainers/
│       │   ├── __init__.py
│       │   ├── pretrain_trainer.py # Pretraining trainer
│       │   ├── finetune_trainer.py # Finetuning trainer
│       │   ├── autoregressive_trainer.py # Autoregressive training
│       │   └── mlm_trainer.py     # Masked Language Model training
│       ├── pretrain.py             # Pretraining entry point
│       ├── finetune.py             # Finetuning entry point
│       └── utils/
│           ├── __init__.py
│           ├── data_utils.py       # Data loading utilities
│           └── model_utils.py      # Model building utilities
└── utils/                          # Utilities (existing)
    ├── __init__.py
    ├── mlp.py
    └── utils.py
```

## Key Components

### 1. Configuration System (`config/`)

- **`base.py`**: Defines configuration dataclasses for all components
- **`cli.py`**: Handles command-line argument parsing
- **`config_manager.py`**: Manages config loading, validation, and merging

**Usage**:
```python
from transaction_transformer.config import ModelConfig, TrainingConfig
from transaction_transformer.config.cli import parse_cli_args

# Parse CLI args
cli_args = parse_cli_args()

# Load and merge config
config_manager = ConfigManager()
config = config_manager.load_config("configs/pretrain.yaml")
config = config_manager.merge_configs(config, cli_args)
```

### 2. Data Pipeline (`data/`)

- **`datasets/`**: Dataset classes for different tasks
- **`loaders/`**: Data loading utilities and custom samplers
- **`preprocessing/`**: Data transformation and tokenization

**Usage**:
```python
from transaction_transformer.data.datasets import TransactionDataset
from transaction_transformer.data.loaders import create_data_loaders

# Create datasets
train_dataset = TransactionDataset(train_config)
val_dataset = TransactionDataset(val_config)

# Create data loaders
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
```

### 3. Training System (`modeling/training/`)

- **`base/`**: Abstract base classes and utilities
- **`trainers/`**: Specialized trainers for different training modes
- **`utils/`**: Training utilities

**Usage**:
```python
from transaction_transformer.modeling.training.trainers import PretrainTrainer
from transaction_transformer.modeling.training.utils import build_model, build_optimizer

# Build model and optimizer
model = build_model(config)
optimizer = build_optimizer(model, config)

# Create trainer
trainer = PretrainTrainer(model, config, device, train_loader, val_loader, optimizer)
trainer.train(num_epochs=config.training.total_epochs)
```

## Training Modes

### 1. Pretraining (`PretrainTrainer`)
- Trains the model from scratch on transaction prediction
- Supports both autoregressive and MLM training
- Saves checkpoints for later finetuning

### 2. Finetuning (`FinetuneTrainer`)
- Loads pretrained weights and finetunes on specific tasks
- Supports fraud detection and other downstream tasks
- Can freeze embedding layers and train only task-specific heads

### 3. Autoregressive Training (`AutoregressiveTrainer`)
- Next token prediction training
- Uses causal masking in transformers
- Suitable for sequence generation tasks

### 4. MLM Training (`MLMTrainer`)
- Masked Language Model training
- Randomly masks tokens and predicts them
- Good for learning robust representations

## Configuration Priority

The system follows this priority order:
1. **CLI arguments** (highest priority)
2. **Config file** (YAML)
3. **Default values** (lowest priority)

**Example**:
```bash
python -m transaction_transformer.main \
    --mode pretrain \
    --config configs/pretrain.yaml \
    --batch-size 64 \
    --learning-rate 1e-4
```

## Checkpoint Management

The new checkpoint system provides:
- **Automatic saving**: Based on steps/epochs
- **Best model tracking**: Saves best model based on validation metrics
- **Resume capability**: Can resume from any checkpoint
- **Cleanup**: Automatically removes old checkpoints

**Usage**:
```python
# Save checkpoint
trainer.save_checkpoint(epoch, step, metrics)

# Load checkpoint
trainer.load_checkpoint("checkpoints/model_epoch_10.pt")

# Resume training
trainer.train(resume_from="checkpoints/latest.pt")
```

## Migration Plan

### Phase 1: Model Components ✅
- [x] Split transformer.py into separate components
- [x] Create component-specific files
- [x] Update imports

### Phase 2: Training System
- [ ] Implement base trainer functionality
- [ ] Implement specialized trainers
- [ ] Add checkpoint management
- [ ] Add metrics tracking

### Phase 3: Configuration System
- [ ] Implement YAML config loading
- [ ] Add CLI argument parsing
- [ ] Implement config validation
- [ ] Add config merging

### Phase 4: Data Pipeline
- [ ] Implement dataset classes
- [ ] Add data loading utilities
- [ ] Implement preprocessing pipeline
- [ ] Add tokenization logic

### Phase 5: Integration
- [ ] Update entry points
- [ ] Add comprehensive testing
- [ ] Update documentation
- [ ] Add examples

## Benefits

1. **Modularity**: Each component has a single responsibility
2. **Testability**: Components can be tested independently
3. **Extensibility**: Easy to add new models, trainers, or datasets
4. **Maintainability**: Smaller, focused files are easier to understand
5. **Reusability**: Components can be reused across different tasks
6. **Configuration**: Flexible configuration system with CLI override
7. **Checkpointing**: Robust save/load functionality with metadata

## Next Steps

1. **Implement skeleton functions**: Fill in the TODO sections in each file
2. **Add tests**: Create comprehensive test suite for each component
3. **Update documentation**: Add detailed docstrings and examples
4. **Create examples**: Add example scripts for each training mode
5. **Performance optimization**: Optimize data loading and training loops

## Usage Examples

### Pretraining
```bash
python -m transaction_transformer.main \
    --mode pretrain \
    --config configs/pretrain.yaml \
    --batch-size 32 \
    --learning-rate 1e-4
```

### Finetuning
```bash
python -m transaction_transformer.main \
    --mode finetune \
    --config configs/finetune.yaml \
    --resume-from data/models/pretrained/checkpoint.pt
```

This new structure provides a solid foundation for building and extending the transaction transformer system while maintaining clean, modular, and maintainable code. 