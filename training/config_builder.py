"""
Configuration building utilities for pretraining.
"""
import torch
from typing import Dict, List, Any, Optional, Union

from configs.config import ModelConfig, TransformerConfig
from models.transformer.transformer_model import TransactionModel


def build_model_config(
    args: Any,
    cat_features: List[str],
    cont_features: List[str],
    enc: Dict[str, Any],
    qparams: Optional[Dict[str, Any]] = None
) -> ModelConfig:
    """
    Build model configuration from arguments and data info.
    
    Args:
        args: Parsed command line arguments
        cat_features: List of categorical feature names
        cont_features: List of continuous feature names  
        enc: Encoding dictionary from preprocessing
        qparams: Quantization parameters for continuous features
        
    Returns:
        ModelConfig instance
    """
    return ModelConfig(
        cat_vocab_sizes={k: len(enc[k]["inv"]) for k in cat_features},
        cont_features=cont_features,
        emb_dropout=args.emb_dropout,
        padding_idx=0,
        total_epochs=args.total_epochs,
        window=args.window,
        stride=args.stride,
        
        ft_config=TransformerConfig(
            d_model=args.ft_d_model,         
            n_heads=args.ft_n_heads,  
            depth=args.ft_depth,   
            ffn_mult=args.ft_ffn_mult, 
            dropout=args.ft_dropout, 
            layer_norm_eps=args.ft_layer_norm_eps,
            norm_first=args.ft_norm_first,
        ),
        
        seq_config=TransformerConfig(
            d_model=args.seq_d_model,
            n_heads=args.seq_n_heads,
            depth=args.seq_depth,
            ffn_mult=args.seq_ffn_mult,
            dropout=args.seq_dropout,
            layer_norm_eps=args.seq_layer_norm_eps,
            norm_first=args.seq_norm_first,
        ),
        
        lstm_config=None,  # Pre-train phase keeps this off
        use_quantized_targets=True,
        cont_vocab_sizes={
            f: qparams[f]['num_bins'] for f in cont_features if f in qparams
        } if qparams else None
    )


def create_model(config: ModelConfig, device: torch.device) -> TransactionModel:
    """
    Create and initialize model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Initialized TransactionModel
    """
    print("Initializing model")
    model = TransactionModel(config).to(device)
    return model


def setup_performance_optimizations(model: TransactionModel, args: Any):
    """
    Apply performance optimizations to model.
    
    Args:
        model: Model to optimize
        args: Arguments containing optimization settings
        
    Returns:
        Optimized model (may be wrapped by torch.compile)
    """
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model for faster training...")
        model = torch.compile(model)
        
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    return model 