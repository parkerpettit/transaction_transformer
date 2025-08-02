# Implementation TODO List

This document outlines what needs to be implemented to complete the Transaction Transformer project according to the project overview rules.

## ✅ Completed Components

### 1. Data Schema & Encoders
- ✅ `FieldSchema` class with proper attributes
- ✅ `CatEncoder` with special IDs (PAD=0, MASK=1, UNK=2, real≥3)
- ✅ `NumBinner` with quantile-based bins
- ✅ Schema utilities (`cat_idx`, `cont_idx`)

### 2. Datasets & Collators
- ✅ `TxnDataset` with sliding window functionality
- ✅ `MLMTabCollator` with proper masking strategy
- ✅ `ARTabCollator` with shifted labels logic
- ✅ Proper dataset output format

### 3. Input Encoding & Embeddings
- ✅ `FrequencyEncoder` for continuous fields (NeRF-style)
- ✅ `EmbeddingLayer` with categorical embeddings and continuous projections
- ✅ Proper NaN handling for masked continuous values
- ✅ Device safety with `.to(self.freqs)`

### 4. Model Architecture
- ✅ `FieldTransformer` for intra-row interactions
- ✅ `RowProjector` for flattening field embeddings
- ✅ `SequenceTransformer` with auto-expanding PE and causal masks
- ✅ `RowExpander` for projecting back to per-field
- ✅ `TransactionPredictionHead` with per-field heads

### 5. Training Framework
- ✅ `BaseTrainer` with main training loop
- ✅ `MLMTrainer` with bidirectional attention
- ✅ `AutoregressiveTrainer` with causal attention
- ✅ Custom loss functions skeleton
- ✅ Wandb integration

## 🔄 Partially Implemented Components

### 1. Custom Cross-Entropy Loss
- ✅ Skeleton implementation with proper structure
- ❌ **TODO**: Implement actual categorical label smoothing logic
- ❌ **TODO**: Implement actual numeric neighborhood smoothing logic
- ❌ **TODO**: Test with real data

### 2. Label Smoothing Strategies
- ✅ `CategoricalLabelSmoothing` class skeleton
- ✅ `NumericNeighborhoodSmoothing` class skeleton
- ❌ **TODO**: Integrate with custom loss function
- ❌ **TODO**: Test smoothing strategies

### 3. Training Scripts
- ✅ `pretrain.py` skeleton with proper structure
- ❌ **TODO**: Load actual schema and data
- ❌ **TODO**: Test end-to-end training

## ❌ Missing Components

### 1. Configuration Management
- ❌ **TODO**: Create proper config files (YAML) for different training modes
- ❌ **TODO**: CLI interface for overriding config settings
- ❌ **TODO**: Config validation and schema integration

### 2. Data Loading & Preprocessing
- ❌ **TODO**: Complete data preprocessing pipeline
- ❌ **TODO**: Save/load schema properly
- ❌ **TODO**: Handle data loading errors gracefully

### 3. Model Checkpointing
- ❌ **TODO**: Implement proper checkpoint saving/loading
- ❌ **TODO**: Save model config and schema with checkpoints
- ❌ **TODO**: Resume training from checkpoints

### 4. Evaluation & Metrics
- ❌ **TODO**: Implement per-field accuracy metrics
- ❌ **TODO**: Implement masked accuracy for MLM
- ❌ **TODO**: Implement bin occupancy histograms
- ❌ **TODO**: Implement F1 score for fraud detection

### 5. Inference Patterns
- ❌ **TODO**: MLM inference for imputation/scoring
- ❌ **TODO**: AR next-row prediction
- ❌ **TODO**: Fast path for last-step inference

### 6. Fraud Detection Model
- ❌ **TODO**: Complete `FraudDetectionModel` implementation
- ❌ **TODO**: Load pretrained embeddings
- ❌ **TODO**: Implement fraud classification head

### 7. Testing & Validation
- ❌ **TODO**: Unit tests for all components
- ❌ **TODO**: Integration tests for training pipeline
- ❌ **TODO**: Validation on real transaction data

## 🚨 Critical Issues to Fix

### 1. Type Annotations
- ❌ **TODO**: Fix linter errors in trainers (model.config access)
- ❌ **TODO**: Add proper type hints throughout codebase
- ❌ **TODO**: Fix import issues

### 2. Data Integration
- ❌ **TODO**: Connect preprocessing output to training pipeline
- ❌ **TODO**: Ensure schema is properly passed through all components
- ❌ **TODO**: Test with actual transaction data

### 3. Loss Function Implementation
- ❌ **TODO**: Complete the custom loss function implementation
- ❌ **TODO**: Test loss computation with real data
- ❌ **TODO**: Verify label smoothing works correctly

## 📋 Implementation Priority

### High Priority (Core Functionality)
1. **Complete custom loss function implementation**
2. **Fix data loading and schema integration**
3. **Test end-to-end training pipeline**
4. **Implement proper checkpointing**

### Medium Priority (Features)
1. **Add evaluation metrics**
2. **Complete fraud detection model**
3. **Add inference patterns**
4. **Implement configuration management**

### Low Priority (Polish)
1. **Add comprehensive testing**
2. **Optimize performance**
3. **Add documentation**
4. **Add CLI interface**

## 🔧 Technical Debt

1. **Type Safety**: Many components need better type annotations
2. **Error Handling**: Add proper error handling throughout pipeline
3. **Logging**: Implement comprehensive logging system
4. **Performance**: Optimize data loading and model inference
5. **Memory**: Handle large datasets efficiently

## 📝 Notes for Implementation

1. **Follow the project outline rules strictly** - especially the special IDs (PAD=0, MASK=1, UNK=2, real≥3)
2. **Use FieldSchema as single source of truth** - no hardcoded IDs or vocab sizes
3. **Implement proper device management** - avoid `torch.as_tensor` on CUDA tensors
4. **Test with small datasets first** - ensure pipeline works before scaling up
5. **Use wandb for all logging** - track experiments properly
6. **Save checkpoints frequently** - include schema and config in checkpoints

## 🎯 Success Criteria

The implementation will be considered complete when:

1. ✅ MLM pretraining works end-to-end with real data
2. ✅ AR pretraining works end-to-end with real data
3. ✅ Custom loss function with label smoothing works correctly
4. ✅ Model can be saved/loaded with checkpoints
5. ✅ Fraud detection model can be trained on pretrained embeddings
6. ✅ All components have proper error handling and logging
7. ✅ Code passes all tests and linting checks 