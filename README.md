# Transaction Transformer

A modern PyTorch implementation of a transformer-based fraud detection system for credit card transactions. This project implements a **two-stage training approach**: unsupervised pre-training on legitimate transactions followed by supervised fine-tuning for fraud classification.

## ✨ Features

- **🚀 Dual Training Modes**: 
  - **Autoregressive (AR)**: Predict next transaction in sequence
  - **Masked Language Model (MLM)**: Predict masked tokens (BERT-style)
- **⚡ Performance Optimizations**: Mixed precision training, gradient checkpointing, torch.compile support
- **🏗️ Modular Architecture**: Field-level and sequence-level transformers
- **📊 Comprehensive Evaluation**: Built-in evaluation metrics and visualization
- **🔧 Easy Configuration**: YAML-based hyperparameter management

## 🏛️ Architecture

### Two-Stage Training Pipeline

1. **Pre-training Phase** (Unsupervised)
   - Train on legitimate transactions only
   - Learn normal transaction patterns
   - Supports both AR and MLM objectives
   - Saves backbone encoder weights

2. **Fine-tuning Phase** (Supervised) 
   - Binary fraud classification
   - Option to freeze/unfreeze backbone
   - Multiple head types: MLP or LSTM
   - Class balancing and weighted sampling

### Model Architecture

```
Input Transaction → Embedding Layer → Field Transformer → Row Projection → Sequence Transformer → Head(s)
                                    (intra-row attn)                      (inter-row attn)
```

- **Field Transformer**: Captures relationships between features within each transaction
- **Sequence Transformer**: Captures temporal patterns across transaction sequences  
- **Multiple Heads**: AR/MLM heads for pre-training, MLP/LSTM heads for fraud detection

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Pre-training

**Autoregressive Mode:**
```bash
python training/pretrain.py --config configs/pretrain.yaml --mode ar
```

**MLM Mode:**
```bash
python training/pretrain.py --config configs/pretrain.yaml --mode masked --mask_prob 0.15
```

### Fine-tuning

```bash
python training/finetune.py --config configs/finetune.yaml --head mlp
```

### Testing

Verify both training modes work:
```bash
python scripts/test_modes.py
```

## ⚙️ Configuration

### Pre-training Config (`configs/pretrain.yaml`)

```yaml
# Training
mode: ar              # 'ar' or 'masked'
mask_prob: 0.15       # MLM masking probability
batch_size: 32
lr: 5.0e-05
total_epochs: 50

# Architecture  
ft_d_model: 72        # Field transformer hidden size
seq_d_model: 1080     # Sequence transformer hidden size
seq_depth: 12         # Number of transformer layers

# Performance
use_mixed_precision: true
gradient_checkpointing: false
compile_model: false
```

### Fine-tuning Config (`configs/finetune.yaml`)

```yaml
# Fine-tuning
head: mlp             # 'mlp' or 'lstm'
freeze_backbone: true
batch_size: 64
lr: 1.0e-04

# Head configuration
mlp_hidden_size: 256
mlp_num_layers: 2

# Data balancing
class_weight: auto
use_weighted_sampler: true
```

## 📁 Project Structure

```
transaction_transformer/
├── configs/              # Configuration files
│   ├── config.py        # Config classes
│   ├── pretrain.yaml    # Pre-training hyperparameters  
│   └── finetune.yaml    # Fine-tuning hyperparameters
├── models/              # Model implementations
│   └── transformer/     # Transformer models
│       └── transformer_model.py
├── training/            # Training scripts
│   ├── pretrain.py      # Pre-training script
│   └── finetune.py      # Fine-tuning script  
├── data/               # Data processing
│   ├── dataset.py      # Dataset classes
│   └── preprocessing.py # Data preprocessing
├── evaluation/         # Evaluation utilities
│   └── evaluate.py     # Evaluation functions
├── utils/              # Utilities
│   ├── utils.py        # General utilities
│   └── masking.py      # MLM masking functions
└── scripts/            # Helper scripts
    └── test_modes.py   # Test both training modes
```

## 🎯 Training Modes

### Autoregressive (AR) Mode
- **Objective**: Predict the next transaction given previous transactions
- **Use Case**: Learn sequential patterns in transaction behavior
- **Loss**: Cross-entropy for categorical features + MSE for continuous features

### Masked Language Model (MLM) Mode  
- **Objective**: Predict masked tokens in transaction sequences
- **Use Case**: Learn bidirectional representations
- **Masking**: 15% of tokens randomly masked
- **Loss**: Only compute loss on masked positions

## 🏃‍♂️ Performance Optimizations

- **Mixed Precision Training**: Reduces memory usage by 50%
- **Gradient Checkpointing**: Trade compute for memory
- **Model Compilation**: Up to 2x speedup with torch.compile
- **Efficient Attention**: Automatic optimization for large sequences

## 📊 Model Performance

The model is designed to:
- Handle sequences of 10-100 transactions
- Process batch sizes of 32-128 efficiently  
- Scale to millions of transactions
- Achieve SOTA fraud detection performance

## 🛠️ Development

### Running Tests
```bash
python scripts/test_modes.py
```

### Key Components
- `TransactionModel`: Main model class with AR/MLM support
- `TxnDataset`: Dataset for loading transaction sequences
- `compute_mlm_loss`: MLM loss computation with masking
- `evaluate`: Comprehensive evaluation metrics

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{transaction_transformer,
  title={Transaction Transformer: Dual-Mode Training for Credit Card Fraud Detection},
  year={2024},
  url={https://github.com/your-repo/transaction_transformer}
}
```

