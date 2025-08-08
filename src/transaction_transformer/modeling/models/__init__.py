"""
Models module for transaction transformer.

Contains model architectures and components.
"""

from .predictor import FeaturePredictionModel
from .embedder import TransformerEmbedder
from .downstream import FraudDetectionModel
from .components import (
    EmbeddingLayer,
    FieldTransformer,
    RowProjector,
    SequenceTransformer,
    FeaturePredictionHead,
    ClassificationHead,
    RowExpander,
)

__all__ = [
    "FeaturePredictionModel",
    "TransformerEmbedder",
    "FraudDetectionModel",
    "EmbeddingLayer",
    "FieldTransformer",
    "RowProjector",
    "SequenceTransformer",
    "FeaturePredictionHead",
    "ClassificationHead",
    "RowExpander",
]
