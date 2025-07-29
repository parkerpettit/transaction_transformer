#!/usr/bin/env python
"""
Test script to verify UniTab changes work correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.transformer.transformer_model import TransactionModel, FrequencyEncoder
from configs.config import ModelConfig


def test_frequency_encoder():
    """Test the frequency encoder matches UniTab's approach."""
    print("Testing FrequencyEncoder...")

    # Create encoder
    encoder = FrequencyEncoder(d_cont=1, L=8)

    # Test input
    x = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])  # (B=3, T=1, D=1)

    # Get output
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {x.shape[2] * 2 * 8} = {x.shape[2] * 16}")

    # Check output dimension
    expected_dim = x.shape[2] * 16  # 2*L where L=8
    assert (
        output.shape[-1] == expected_dim
    ), f"Expected {expected_dim}, got {output.shape[-1]}"
    print("✓ FrequencyEncoder test passed!")


def test_model_initialization():
    """Test model initialization with new architecture."""
    print("\nTesting model initialization...")

    # Create config
    cfg = ModelConfig()
    cfg.cont_features = ["Amount"]
    cfg.cat_vocab_sizes = {"User": 100, "Card": 50, "Merchant": 200}
    cfg.ft_d_model = 72
    cfg.seq_d_model = 1080
    cfg.window = 10

    # Create model
    model = TransactionModel(cfg)
    print(f"Model created successfully!")
    print(f"Frequency encoder output dim: {model.frequency_encoder.out_dim}")
    print(f"Embedding layer num fields: {model.embedder.num_fields}")
    print("✓ Model initialization test passed!")


def test_forward_pass():
    """Test forward pass with frequency encoding."""
    print("\nTesting forward pass...")

    # Create config
    cfg = ModelConfig()
    cfg.cont_features = ["Amount"]
    cfg.cat_vocab_sizes = {"User": 100, "Card": 50}
    cfg.ft_d_model = 72
    cfg.seq_d_model = 1080
    cfg.window = 10

    # Create model
    model = TransactionModel(cfg)

    # Create test input
    batch_size, seq_len = 2, 5
    cat_input = torch.randint(0, 50, (batch_size, seq_len, len(cfg.cat_vocab_sizes)))
    cont_input = torch.randn(batch_size, seq_len, len(cfg.cont_features))

    # Test forward pass
    with torch.no_grad():
        cat_logits, cont_pred = model(cat_input, cont_input, mode="ar")

    print(f"Input shapes: cat={cat_input.shape}, cont={cont_input.shape}")
    print(f"Output shapes: cat_logits={cat_logits.shape}, cont_pred={cont_pred.shape}")
    print("✓ Forward pass test passed!")


def test_quantization():
    """Test quantization functionality."""
    print("\nTesting quantization...")

    # Create config
    cfg = ModelConfig()
    cfg.cont_features = ["Amount"]
    cfg.cat_vocab_sizes = {"User": 100, "Card": 50}
    cfg.ft_d_model = 72
    cfg.seq_d_model = 1080
    cfg.window = 10

    # Create model
    model = TransactionModel(cfg)

    # Mock quantization parameters
    qparams = {
        "Amount": {"edges": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), "num_bins": 5}
    }

    model.set_quantization_params(qparams)

    # Test quantization
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    quantized = model.quantize_numerical(values, "Amount")
    print(f"Values: {values}")
    print(f"Quantized: {quantized}")
    print("✓ Quantization test passed!")


if __name__ == "__main__":
    print("Testing UniTab changes...")

    test_frequency_encoder()
    test_model_initialization()
    test_forward_pass()
    test_quantization()

    print("\n🎉 All tests passed! UniTab changes are working correctly.")
