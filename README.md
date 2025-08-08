# Transaction Transformer

A Transformer-based architecture for credit-card fraud detection. Pretrain on next-transaction prediction, then fine-tune for binary fraud classification. Supports both autoregressive and BERT-style modes.

## Overview

This project implements a unified tabular transformer architecture (UniTTab) for fraud detection on transaction data. The model uses a two-stage approach:

1. **Pretraining**: Learn representations using either:
   - **MLM (BERT-style)**: Masked token modeling on tabular time series
   - **AR (Autoregressive)**: Next-row prediction

2. **Fine-tuning**: Binary fraud classification using the pretrained representations

## Architecture

### Core Components

- **Field Transformer**: Processes intra-row interactions between features
- **Sequence Transformer**: Captures inter-row temporal dependencies  
- **RowProjector/RowExpander**: Standardizes different row types for the sequence transformer
- **Per-field Heads**: Separate classifiers for each feature type

### Input Encoding

- **Categorical Features**: Standard embedding layers with special tokens (PAD=0, MASK=1, UNK=2)
- **Continuous Features**: Frequency encoding with sin/cos projections
- **Masking**: Learned mask vectors for both categorical and continuous features

### Training Modes

- **MLM Mode**: Bidirectional attention, masked positions for prediction
- **AR Mode**: Causal attention, predict next transaction from history

## Installation

```bash
# Clone the repository
git clone https://github.com/parkerpettit/transaction_transformer.git
cd transaction_transformer

# Install the package
pip install -e .

```

## Data Preparation

### 1. Raw Data Format
Download the TabFormer credit card dataset from:

https://github.com/IBM/TabFormer/tree/main/data/credit_card

The file you need is named `card_transaction.v1.csv` (this is the default filename). Place this file in `data/raw/`. The code expects this exact path and filename, although this can be changed in the config files.

After preprocessing, the dataset will include the following columns:

- **Categorical**: User, Card, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Year, Month, Day, Hour
- **Continuous**: Amount
- **Target**: is_fraud (binary)

### 2. Run Preprocessing

```bash
# After installing with pip install -e, you can use the direct command:
run_preprocessing

# Or use the full module path:
python -m transaction_transformer.data.preprocessing.run_preprocessing
```

This will:
- Split data into train/val/test sets
- Create separate "legit" and "full" datasets
- Fit encoders and scalers on training data only
- Build quantile binners for continuous features
- Save processed datasets to `data/processed/`

## Training

### Configuration

The project uses YAML configuration files for all settings:

- `src/transaction_transformer/config/pretrain.yaml`: Pretraining settings
- `src/transaction_transformer/config/finetune.yaml`: Fine-tuning settings

**All defaults are handled in config files** - the CLI is only used to override settings for quick testing. You can edit the YAML files directly to change any settings.

Key configuration options:
- Model architecture (embedding dim, transformer layers, etc.)
- Training parameters (batch size, learning rate, epochs)
- Data settings (window size, stride, masking probabilities)
- Logging and evaluation metrics

### 1. Pretraining

```bash
# After installing with pip install -e, you can use the direct commands:

# Simple approach - edit the YAML file and run:
# Edit src/transaction_transformer/config/pretrain.yaml to set model_type: "mlm" or "ar"
pretrain

# Or use CLI arguments to override settings:
# MLM pretraining (BERT-style)
pretrain --config src/transaction_transformer/config/pretrain.yaml --model.training.model_type mlm

# AR pretraining (autoregressive)
pretrain --config src/transaction_transformer/config/pretrain.yaml --model.training.model_type ar

# Or use the full module paths:
python -m transaction_transformer.modeling.training.pretrain \
    --config src/transaction_transformer/config/pretrain.yaml \
    --model.training.model_type mlm
```

### 2. Fine-tuning

```bash
# After installing with pip install -e, you can use the direct command:

# Simple approach - edit the YAML file and run:
# Edit src/transaction_transformer/config/finetune.yaml to set pretrain_checkpoint_dir
finetune

# Or use CLI arguments to override settings:
finetune --config src/transaction_transformer/config/finetune.yaml --model.pretrain_checkpoint_dir data/models/pretrained

# Or use the full module path:
python -m transaction_transformer.modeling.training.finetune \
    --config src/transaction_transformer/config/finetune.yaml \
    --model.pretrain_checkpoint_dir data/models/pretrained
```

### Evaluation

Evaluation happens during training (validation metrics per epoch) and is logged to W&B.
To select a model, use the W&B artifact UI to compare `pretrain-<runid>` or `finetune-<runid>` versions and download the desired alias (e.g., `:best`).

## Project Structure

```
transaction_transformer/
├── src/transaction_transformer/
│   ├── config/                 # Configuration files
│   │   ├── pretrain.yaml      # Pretraining settings
│   │   ├── finetune.yaml      # Fine-tuning settings
│   │   └── config.py          # Configuration management
│   ├── data/                  # Data processing
│   │   ├── preprocessing/     # Data preprocessing pipeline
│   │   ├── dataset.py        # Dataset classes
│   │   └── collator.py       # Data collation for training
│   ├── modeling/             # Model architecture
│   │   ├── models/           # Model implementations
│   │   └── training/         # Training scripts and trainers
│   └── utils/               # Utility functions
├── data/                    # Data directory
│   ├── raw/                # Raw input data
│   ├── processed/          # Preprocessed datasets
│   └── models/             # Saved model checkpoints
└── configs/                # Additional configuration files
```

## Key Features

### Unified Architecture
- Single model handles both categorical and continuous features
- Frequency encoding for continuous features (no discretization)
- Unified cross-entropy loss with label smoothing for finetuning

### Flexible Training
- Interchangeable MLM and AR pretraining modes
- Configurable masking strategies
- Support for mixed precision training

### Comprehensive Logging
- Weights & Biases integration for experiment tracking
- Per-field accuracy and loss monitoring
- Gradient and parameter logging

### Data Safety
- No data leakage: encoders/scalers fit on training data only
- Proper train/val/test splits
- Separate processing for fraud and non-fraud datasets

## Default Configuration

### Model Architecture
- **Field Transformer**: 1 layer, 8 heads, 72 dim
- **Sequence Transformer**: 12 layers, 12 heads, 1080 dim
- **Embedding**: 72 dim with frequency encoding (L=8)

### Training Parameters
- **Pretraining**: 5 epochs, batch size 64, learning rate 1e-4
- **Fine-tuning**: 10 epochs, batch size 128, learning rate 1e-4
- **Masking**: 15% field masking, 10% row masking

### Data Processing
- **Window**: 10 transactions per sequence
- **Binning**: 100 quantile bins per continuous feature
- **Special tokens**: PAD=0, MASK=1, UNK=2

## Usage Examples

### Quick Start

1. **Prepare your data**:
   ```bash
   # Place your CSV file in data/raw/card_transaction.v1.csv
   run_preprocessing
   ```

2. **Pretrain the model**:
   ```bash
   pretrain
   ```

3. **Fine-tune for fraud detection**:
   ```bash
   finetune
   ```


### Custom Configuration

You can override default settings in several ways:

**Option 1: Edit YAML files (recommended)**
```bash
# Edit the configuration files directly:
# - src/transaction_transformer/config/pretrain.yaml
# - src/transaction_transformer/config/finetune.yaml
# Then run without CLI args:
pretrain
finetune
```

**Option 2: Use CLI arguments**
```bash
# After installing with pip install -e, you can use the direct command:
pretrain --model.training.batch_size 32 --model.training.learning_rate 5e-5 --model.sequence_transformer.depth 6

# Or use the full module path:
python -m transaction_transformer.modeling.training.pretrain \
    --model.training.batch_size 32 \
    --model.training.learning_rate 5e-5 \
    --model.sequence_transformer.depth 6
```

## Monitoring and Logging

The project integrates with Weights & Biases for experiment tracking:

- **Pretraining**: Project "feature-predictor"
- **Fine-tuning**: Project "fraud-detection"

Tracked metrics include:
- Total loss and per-field losses
- Masked accuracy (MLM mode)
- Fraud detection metrics (precision, recall, F1, AUC)
- Gradient norms and learning rates
- **Automatic evaluation**: Validation metrics are computed during training and the best model is saved based on validation loss





