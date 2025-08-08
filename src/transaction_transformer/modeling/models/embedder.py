import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.components import (
    EmbeddingLayer,
    FieldTransformer,
    RowProjector,
    SequenceTransformer,
)
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema


# -------------------------------------------------------------------------------------- #
#  Transformer Embedder                                                                           #
# -------------------------------------------------------------------------------------- #
class TransformerEmbedder(nn.Module):
    """Tabular-sequence model for embedding the tabular data. Returns (B, L, M) where M is the embedding dimension.
    The embedding model is used to embed the tabular data into a latent space. This is then used to train the downstream model.

    Parameters
    ----------
    config : ModelConfig
        Configuration for the model.
    schema : FieldSchema
        Schema for the model.
    """

    # --------------------------------------------------------------------- #
    def __init__(self, config: ModelConfig, schema: FieldSchema):
        """
        Parameters
        ----------
        config      : ModelConfig
        """
        super().__init__()
        self.config: ModelConfig = config

        # -- Embedding & field transformer ----------------------------------
        self.embedder = EmbeddingLayer(
            schema=schema,
            config=config,
        )
        self.causal = (
            config.training.model_type == "ar"
        )  # causal masking for AR training, mlm is bidirectional

        self.field_tf = FieldTransformer(config.field_transformer)

        # -- Row projection (flatten K*D -> M) -------------------------------
        self.row_proj = RowProjector(config, schema)

        # -- Sequence transformer -------------------------------------------
        self.seq_tf = SequenceTransformer(
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
        init_embeddings = self.embedder(cat, cont)  # (B*L, K, D)
        embeddings = self.field_tf(init_embeddings)  # (B*L, K, D)
        row_embeddings = self.row_proj(
            embeddings, row_type=row_type, B=B, L=L
        )  # (B, L, M)
        seq_embeddings = self.seq_tf(row_embeddings)  # (B, L, M)
        return seq_embeddings  # (B, L, M)
