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

    def load_pretrained_embedding_backbone(self, checkpoint_path: str):
        """
        Load a pretrained backbone from a pretraining checkpoint.

        Args:
            checkpoint_path: Path to the pretrained model checkpoint
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        weights = torch.load(checkpoint_path, weights_only=False)
        embedding_state_dict = {
            k[len("backbone.") :]: v
            for k, v in weights["model_state_dict"].items()
            if k.startswith("backbone.")
        }
        print(embedding_state_dict.keys())
        # Load the weights into our backbone
        self.backbone.load_state_dict(embedding_state_dict)

        # Freeze the embedding model if specified
        if self.freeze_embedding:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Clean up the temporary model
        del weights

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
        last_embedding = embeddings[:, -1, :]  # (B, M)
        # Pass through the classification head
        logits = self.head(last_embedding)  # (B, 1) -> squeeze to (B,)
        return logits.squeeze(-1)  # (B,)

        # TODO change so that it puts all B, L, M embeddings through instead of just the last one, because lstm head needs all of them
        # and then change the head to slice off the last embedding and pass that through the mlp
