# UniTab Architecture Changes

This document summarizes the changes made to align your codebase with the UniTab paper architecture while maintaining your autoregressive training capability.

## Key Changes Made

### 1. **Frequency-Based Encoding** (UniTab Style)
- **Replaced**: `FourierNumericEncoder` -> `FrequencyEncoder`
- **Implementation**: Matches UniTab's gamma(v) = (sin(2^0*pi*v), cos(2^0*pi*v), ..., sin(2^(L-1)*pi*v), cos(2^(L-1)*pi*v))
- **Parameters**: L=8, frequencies = [1, 2, 4, 8, 16, 32, 64, 128]
- **Output**: 16-dimensional encoding per continuous feature (2*L)

### 2. **Embedding Layer Updates**
- **Updated**: `EmbeddingLayer` to handle frequency-encoded continuous features
- **Change**: Continuous projections now take 16-dim input instead of 1-dim
- **Maintained**: Categorical embeddings remain unchanged

### 3. **Quantization Support**
- **Added**: Quantization parameters loading from preprocessing
- **Added**: `quantize_numerical()` method for training targets
- **Added**: `neighborhood_label_smoothing()` for numerical features
- **Added**: `masked_token_loss()` method for unified training

### 4. **Training Mode Support**
- **Added**: Support for both autoregressive (`ar`) and masked token (`masked`) training
- **Updated**: `pretrain.py` to handle both modes via `--mode` argument
- **Maintained**: Your existing autoregressive training capability

### 5. **Architecture Simplifications**
- **Kept**: Row projection since you only have one row type
- **Removed**: Type-dependent projections (not needed for single row type)
- **Maintained**: Your hierarchical transformer structure

## Files Modified

### Core Model Changes
- `models/transformer/transformer_model.py`
  - Replaced `FourierNumericEncoder` with `FrequencyEncoder`
  - Updated `EmbeddingLayer` for frequency encoding
  - Added quantization support methods
  - Added masked token training capability

### Training Changes
- `training/pretrain.py`
  - Added `--mode` argument for training mode selection
  - Added masked batch creation function
  - Updated training loop to handle both modes
  - Added quantization parameter loading

### Data Processing
- `data/preprocessing.py` (already had quantization)
  - Quantization parameters are already implemented
  - Saves quantization parameters with processed data

## Usage

### Autoregressive Training (Your Original Method)
```bash
python training/pretrain.py --config configs/pretrain.yaml --mode ar
```

### Masked Token Training (UniTab Method)
```bash
python training/pretrain.py --config configs/pretrain.yaml --mode masked
```

## Key Differences from UniTab Paper

### What We Kept (Your Advantages)
1. **Autoregressive Training**: You can still train autoregressively to predict next transactions
2. **Multiple Heads**: Your flexible head system (MLP, LSTM, autoregressive)
3. **Causal Masking**: Your sequence transformer uses causal masking for autoregressive training
4. **Row Projection**: Kept since you only have one row type

### What We Added (UniTab Features)
1. **Frequency Encoding**: Replaced Fourier with UniTab's frequency-based encoding
2. **Quantization**: Added quantization for numerical feature targets
3. **Neighborhood Label Smoothing**: For numerical features during training
4. **Masked Token Training**: Alternative training mode
5. **Unified Loss**: Single loss function for both categorical and numerical features

## Testing

Run the test script to verify all changes work:
```bash
python scripts/test_unittab_changes.py
```

## Benefits

1. **Better Numerical Representation**: Frequency encoding should improve numerical feature learning
2. **Flexible Training**: You can experiment with both autoregressive and masked token training
3. **Unified Approach**: Single loss function for all feature types
4. **Backward Compatibility**: Your existing autoregressive training still works
5. **Future Extensibility**: Easy to add variable row types later if needed

## Next Steps

1. **Compare Training Modes**: Run both `--mode ar` and `--mode masked` to see which performs better
2. **Hyperparameter Tuning**: Adjust frequency encoding parameters (L=8) if needed
3. **Add Variable Row Types**: If you get data with different transaction types, add type-dependent projections
4. **Advanced Masking**: Implement row masking and timestamp masking strategies

## Architecture Summary

Your updated architecture now matches UniTab's key innovations:

```
Input -> Frequency Encoding -> Embedding -> Field Transformer ->
Row Projection -> Sequence Transformer -> Heads (AR/LSTM/MLP)
```

This maintains your hierarchical structure while incorporating UniTab's frequency-based encoding and quantization approach. 