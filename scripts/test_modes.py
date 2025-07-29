#!/usr/bin/env python
"""
Test script to verify that both AR and MLM training modes work correctly.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import ModelConfig, TransformerConfig
from models.transformer.transformer_model import TransactionModel
from utils.masking import create_mlm_mask, apply_categorical_masking, apply_continuous_masking, compute_mlm_loss

def test_model_modes():
    """Test that the model works in both AR and MLM modes."""
    print("Testing Transaction Transformer modes...")
    
    # Create a simple config for testing
    config = ModelConfig(
        cat_vocab_sizes={'User': 100, 'Card': 50, 'MCC': 20},
        cont_features=['Amount'],
        emb_dropout=0.1,
        clf_dropout=0.1,
        padding_idx=0,
        total_epochs=1,
        window=5,
        stride=1,
        ft_config=TransformerConfig(
            d_model=32,
            n_heads=2,
            depth=1,
            ffn_mult=2,
            dropout=0.1
        ),
        seq_config=TransformerConfig(
            d_model=64,
            n_heads=2,
            depth=2,
            ffn_mult=2,
            dropout=0.1
        )
    )
    
    # Create model
    model = TransactionModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create sample data
    batch_size = 4
    seq_len = 5
    num_cat_features = 3
    num_cont_features = 1
    
    cat_input = torch.randint(1, 10, (batch_size, seq_len, num_cat_features), device=device)
    cont_input = torch.randn(batch_size, seq_len, num_cont_features, device=device)
    
    print(f"Input shapes: cat={cat_input.shape}, cont={cont_input.shape}")
    
    # Test AR mode
    print("\n1. Testing Autoregressive (AR) mode...")
    try:
        model.eval()
        with torch.no_grad():
            cat_logits, cont_pred = model(cat_input, cont_input, mode="ar")
            print(f"   AR outputs: cat_logits={cat_logits.shape}, cont_pred={cont_pred.shape}")
            
            # Check that AR outputs are for the last timestep only
            expected_cat_size = sum(config.cat_vocab_sizes.values())
            assert cat_logits.shape == (batch_size, expected_cat_size), f"Expected {(batch_size, expected_cat_size)}, got {cat_logits.shape}"
            assert cont_pred.shape == (batch_size, len(config.cont_features)), f"Expected {(batch_size, len(config.cont_features))}, got {cont_pred.shape}"
            print("   ✓ AR mode works correctly")
    except Exception as e:
        print(f"   ✗ AR mode failed: {e}")
        return False
    
    # Test MLM mode
    print("\n2. Testing Masked Language Model (MLM) mode...")
    try:
        model.eval()
        with torch.no_grad():
            cat_logits, cont_pred = model(cat_input, cont_input, mode="masked")
            print(f"   MLM outputs: cat_logits={cat_logits.shape}, cont_pred={cont_pred.shape}")
            
            # Check that MLM outputs are for all timesteps
            expected_cat_size = sum(config.cat_vocab_sizes.values())
            assert cat_logits.shape == (batch_size, seq_len, expected_cat_size), f"Expected {(batch_size, seq_len, expected_cat_size)}, got {cat_logits.shape}"
            assert cont_pred.shape == (batch_size, seq_len, len(config.cont_features)), f"Expected {(batch_size, seq_len, len(config.cont_features))}, got {cont_pred.shape}"
            print("   ✓ MLM mode works correctly")
    except Exception as e:
        print(f"   ✗ MLM mode failed: {e}")
        return False
    
    # Test MLM training loop
    print("\n3. Testing MLM training with masking...")
    try:
        model.train()
        
        # Create mask
        mask = create_mlm_mask(batch_size, seq_len, mask_prob=0.3, device=device)
        print(f"   Created mask with shape: {mask.shape}, {mask.sum().item()}/{mask.numel()} positions masked")
        
        # Apply masking
        cat_masked = apply_categorical_masking(cat_input, mask, config.cat_vocab_sizes)
        cont_masked = apply_continuous_masking(cont_input, mask)
        
        # Forward pass with masked input
        cat_logits, cont_pred = model(cat_masked, cont_masked, mode="masked")
        
        # Compute MLM loss
        loss, cat_loss, cont_loss = compute_mlm_loss(
            cat_logits, cont_pred, cat_input, cont_input, mask,
            config.cat_vocab_sizes, list(config.cat_vocab_sizes.keys())
        )
        
        print(f"   MLM losses: total={loss.item():.4f}, cat={cat_loss.item():.4f}, cont={cont_loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("   ✓ MLM training works correctly")
        
    except Exception as e:
        print(f"   ✗ MLM training failed: {e}")
        return False
    
    # Test AR training loop
    print("\n4. Testing AR training...")
    try:
        model.train()
        
        # Forward pass
        cat_logits, cont_pred = model(cat_input, cont_input, mode="ar")
        
        # Create targets (next timestep)
        cat_targets = cat_input[:, -1, :]  # (B, C)
        cont_targets = cont_input[:, -1, :]  # (B, F)
        
        # Compute AR loss (simplified)
        criterion_cat = nn.CrossEntropyLoss()
        criterion_cont = nn.MSELoss()
        
        # Split categorical logits by feature and compute loss
        logit_splits = list(config.cat_vocab_sizes.values())
        cat_logit_list = torch.split(cat_logits, logit_splits, dim=-1)
        
        cat_loss = 0
        for i, feat_logits in enumerate(cat_logit_list):
            cat_loss += criterion_cat(feat_logits, cat_targets[:, i])
        cat_loss /= len(cat_logit_list)
        
        cont_loss = criterion_cont(cont_pred, cont_targets)
        total_loss = cat_loss + cont_loss
        
        print(f"   AR losses: total={total_loss.item():.4f}, cat={cat_loss.item():.4f}, cont={cont_loss.item():.4f}")
        
        # Test backward pass
        total_loss.backward()
        print("   ✓ AR training works correctly")
        
    except Exception as e:
        print(f"   ✗ AR training failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Both AR and MLM modes work correctly.")
    return True

if __name__ == "__main__":
    success = test_model_modes()
    if not success:
        sys.exit(1)