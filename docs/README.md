# Transaction Transformer Documentation

## Overview

This project implements an autoregressive Transformer for credit-card fraud detection using a two-stage pipeline:

1. **Pretraining**: Self-supervised learning on next-transaction prediction
2. **Finetuning**: Binary classification for fraud detection

## Project Structure

```
transaction_transformer/
├── data/                   # Data files
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Clean, structured data
│   ├── external/          # Data from third party sources
│   ├── interim/           # Intermediate data
│   └── models/            # Trained and serialized models
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
├── reports/               # Generated analysis as HTML, PDF, LaTeX, etc.
├── src/                   # Source code
│   └── transaction_transformer/
│       ├── data/          # Data loading and processing
│       ├── features/      # Feature engineering
│       ├── models/        # Model implementations and training
│       ├── utils/         # Utility functions
│       └── visualization/ # Visualization and EDA
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Quick Start

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Run pretraining**:
   ```bash
   make train-pretrain
   ```

3. **Run finetuning**:
   ```bash
   make train-finetune
   ```

4. **Run evaluation**:
   ```bash
   make evaluate
   ```

## Configuration

All training parameters are configured via YAML files in the `configs/` directory:

- `configs/pretrain.yaml`: Pretraining configuration
- `configs/finetune.yaml`: Finetuning configuration
- `configs/sweep.yaml`: Hyperparameter sweep configuration

## Model Architecture

The project implements:

- **Transformer Model**: Autoregressive transformer for sequence modeling
- **LightGBM Baseline**: Gradient boosting baseline for comparison
- **Flexible Heads**: MLP or LSTM classifiers on frozen/unfrozen transformer

## Data Pipeline

1. **Raw Data**: IBM TabFormer dataset (24M transactions)
2. **Preprocessing**: Feature engineering and encoding
3. **Training**: Two-stage pipeline (pretrain → finetune)
4. **Evaluation**: ROC/PR curves, confusion matrices

## Development

- **Format code**: `make format`
- **Run tests**: `make test`
- **Lint code**: `make lint`
- **Clean build**: `make clean` 