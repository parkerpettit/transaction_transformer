import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.components import (
    EmbeddingLayer,
    FieldTransformer,
    RowProjector,
    SequenceTransformer,
)
from transaction_transformer.data.preprocessing.schema import FieldSchema


# -------------------------------------------------------------------------------------- #
#  Backbone (shared embedding + transformers)                                            #
# -------------------------------------------------------------------------------------- #
class Backbone(nn.Module):
    """Tabular-sequence backbone that maps per-field inputs to sequence embeddings.
    Returns (B, L, M) where M is the sequence embedding dimension.

    Parameters
    ----------
    config : ModelConfig
        Configuration for the model.
    schema : FieldSchema
        Schema for the model.
    """

    # --------------------------------------------------------------------- #
    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.config: ModelConfig = config

        # -- Embedding & field transformer ----------------------------------
        self.field_embedding = EmbeddingLayer(
            schema=schema,
            config=config,
        )
        self.causal = config.training.model_type == "ar"  # causal masking for AR training; MLM is bidirectional

        self.field_transformer = FieldTransformer(config.field_transformer)

        # -- Row projection (flatten K*D -> M) -------------------------------
        self.row_projector = RowProjector(config, schema)

        # -- Sequence transformer -------------------------------------------
        self.sequence_transformer = SequenceTransformer(
            config.sequence_transformer, causal=self.causal
        )

    # --------------------------------------------------------------------- #
    def forward(
        self,
        cat: LongTensor,  # (B, L, C)
        cont: Tensor,  # (B, L, F)
        row_type: int = 0,  # Index of the row type to project the embeddings into
    ):
        # (K = C + F)
        B, L, _ = cat.shape
        init_embeddings = self.field_embedding(cat, cont)  # (B*L, K, D)
        embeddings = self.field_transformer(init_embeddings)  # (B*L, K, D)
        row_embeddings = self.row_projector(
            embeddings, row_type=row_type, B=B, L=L
        )  # (B, L, M)
        seq_embeddings = self.sequence_transformer(row_embeddings)  # (B, L, M)
        return seq_embeddings  # (B, L, M)


