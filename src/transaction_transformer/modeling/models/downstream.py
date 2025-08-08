import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.embedder import TransformerEmbedder
from transaction_transformer.modeling.models.components import ClassificationHead
from transaction_transformer.modeling.models.predictor import FeaturePredictionModel
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema
from transaction_transformer.modeling.training.base.checkpoint_manager import (
    CheckpointManager,
)
import torch
import numpy as np
from typing import cast


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

        # Initialize the embedding model (this will be loaded with pretrained weights)
        self.embedding_model = TransformerEmbedder(config, schema)

        # Initialize the classification head
        self.classification_head = ClassificationHead(config)

        # Flag to control whether to freeze the embedding model
        self.freeze_embedding = config.freeze_embedding

    def load_pretrained_embedding_model(self, checkpoint_path: str):
        """
        Load a pretrained FeaturePredictionModel and extract just the embedding part.

        Args:
            checkpoint_path: Path to the pretrained model checkpoint
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        weights = torch.load(checkpoint_path, weights_only=False)
        # Extract only the transaction embedding submodule weights and strip the prefix
        prefix = "transaction_embedding_model."
        raw_state_dict = weights["model_state_dict"]
        embedding_state_dict: dict[str, torch.Tensor] = {}
        for k, v in raw_state_dict.items():
            if k.startswith(prefix):
                # Remove the prefix so keys match this module's state_dict
                new_key = k[len(prefix) :]
                embedding_state_dict[new_key] = v
        print(
            f"Loaded {len(embedding_state_dict)} embedding keys from pretrained checkpoint"
        )

        # 1) Load all matching-shaped weights EXCEPT categorical embeddings (handle those manually)
        filtered: dict[str, torch.Tensor] = {}
        current_state = self.embedding_model.state_dict()
        for k, v in embedding_state_dict.items():
            if k.startswith("embedder.cat_emb."):
                continue  # handle below
            if k in current_state and current_state[k].shape == v.shape:
                filtered[k] = v
        missing_before = set(current_state.keys()) - set(filtered.keys())
        self.embedding_model.load_state_dict(filtered, strict=False)

        # 2) If pretraining schema available, remap categorical embeddings token-wise
        pre_schema = weights.get("schema", None)
        if pre_schema is not None:
            # Access current embedding layer
            emb_layer = self.embedding_model.embedder
            cat_names = emb_layer.cat_features
            for i, (emb_mod, feat_name) in enumerate(zip(emb_layer.cat_emb, cat_names)):
                # Get current weight tensor (start from its random init)
                w_cur = cast(torch.Tensor, emb_mod.weight.data)  # (V_new, D)
                key_old = f"{prefix}embedder.cat_emb.{i}.weight"
                # If old weights exist
                if key_old in raw_state_dict:
                    w_old = cast(torch.Tensor, raw_state_dict[key_old])  # (V_old, D)
                    # Build token->index maps using encoder inv arrays
                    new_inv = emb_layer.cat_encoders[feat_name].inv  # np.ndarray
                    old_inv = pre_schema.cat_encoders[feat_name].inv  # type: ignore[attr-defined]
                    # Ensure numpy list to avoid typing issues
                    old_list = (
                        old_inv.tolist()
                        if hasattr(old_inv, "tolist")
                        else list(old_inv)
                    )
                    old_index = {tok: idx for idx, tok in enumerate(old_list)}
                    # Copy rows where token exists in old
                    new_list = (
                        new_inv.tolist()
                        if hasattr(new_inv, "tolist")
                        else list(new_inv)
                    )
                    for new_id, tok in enumerate(new_list):
                        old_id = old_index.get(tok, None)
                        if old_id is not None and old_id < int(w_old.size(0)):
                            if new_id < int(w_cur.size(0)):
                                w_cur[new_id].copy_(w_old[old_id])
                # else: keep current initialization
        else:
            # Fallback: skip cat embeddings if schema missing (sizes may differ)
            pass

        # Freeze the embedding model if specified
        if self.freeze_embedding:
            for param in self.embedding_model.parameters():
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
        embeddings = self.embedding_model(cat, cont, row_type)  # (B, L, M)
        last_embedding = embeddings[:, -1, :]  # (B, M)
        # Pass through the classification head
        logits = self.classification_head(last_embedding)  # (B, 1) -> squeeze to (B,)
        return logits.squeeze(-1)  # (B,)
