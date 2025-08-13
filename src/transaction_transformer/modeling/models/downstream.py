import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.backbone import Backbone
from transaction_transformer.modeling.models.components import (
    ClassificationHead,
    LSTMHead,
)
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.modeling.training.base.checkpoint_manager import (
    CheckpointManager,
)
import torch


class FraudDetectionModel(nn.Module):
    """
    Fraud detection model. Takes a pretrained transaction embedding model and a fraud classification head.
    Returns a (B,) logits. Can be set to finetune end-to-end or train the fraud classification head alone
    on top of the frozen transaction embedding model. Controlled by the freeze_embedding flag in the config.
    """

    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.config = config
        self.schema = schema

        # Initialize the backbone (this will be loaded with pretrained weights)
        self.backbone = Backbone(config, schema)

        # Initialize the classification head

        self.head = ClassificationHead(config)
        # uncomment to use lstm head instead
        # self.head = LSTMHead(config)

        # Flag to control whether to freeze the embedding model
        self.freeze_embedding = config.freeze_embedding


    def forward(self, cat: LongTensor, cont: Tensor, row_type: int = 0):
        """
        Forward pass for fraud detection.

        Args:
            cat: Categorical features (B, L, C)
            cont: Continuous features (B, L, F)
            row_type: Row type (default 0 for single row type)

        Returns:
            logits: (B,) logits for fraud classification
        """
        # Get embeddings from the pretrained embedding model
        embeddings = self.backbone(cat, cont, row_type)  # (B, L, M)
        # Pass through the classification head
        logits = self.head(embeddings)  # (B, 1) -> squeeze to (B,)
        return logits.squeeze(-1)  # (B,)
