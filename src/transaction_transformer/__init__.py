"""
Feature Prediction Transformer Package

An autoregressive Transformer for tabular data prediction.
Pretrain on next-row prediction, then fine-tune for downstream tasks.
"""

__version__ = "0.1.0"
__author__ = "Parker Pettit"
__email__ = "ppettit@mit.edu"

# Import main classes and functions from submodules
from .data.preprocessing import (
    FieldSchema,
    get_encoders,
    get_scaler,
    build_quantile_binner,
    normalize,
    encode_df,
    preprocess,
)
from .data import TxnDataset, MLMTabCollator, ARTabCollator, BaseTabCollator

from .modeling.models import (
    FeaturePredictionModel,
    TransformerEmbedder,
    FraudDetectionModel,
)

from .modeling.training import main

from .config.config import ModelConfig

# Convenience imports for common use cases
__all__ = [
    # Data preprocessing
    "FieldSchema",
    "get_encoders",
    "get_scaler",
    "build_quantile_binner",
    "normalize",
    "encode_df",
    "preprocess",
    # Datasets and collators
    "TxnDataset",
    "MLMTabCollator",
    "ARTabCollator",
    "BaseTabCollator",
    # Models
    "FeaturePredictionModel",
    "TransformerEmbedder",
    "FraudDetectionModel",
    # Training
    "main",
    # Config
    "ModelConfig",
]
