import torch
import torch.nn as nn
from torch import Tensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.data.preprocessing.schema import FieldSchema


class RowProjector(nn.Module):
    """
    Project the field-level embeddings into a row-level embedding. Supports different row types,
    each with its own linear projection. Row types are defined in the ModelConfig. For the TabFormer
    dataset, there is only one row type. However, some datasets may have multiple row types, which is why
    this is implemented. For example, in some datasets, there may be a row schema for the transaction data and a
    row schema for the customer data.

    Projects (B*L, K, D) -> (B, L, M) where M is the hidden dimension of the sequence transformer.
    """

    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.row_types = config.row_types
        num_fields = len(schema.cat_features) + len(schema.cont_features)
        self.row_projs = nn.ModuleList(
            [
                nn.Linear(
                    num_fields * config.field_transformer.d_model,
                    config.sequence_transformer.d_model,
                )
                for _ in range(self.row_types)
            ]
        )
        self.M = config.sequence_transformer.d_model

    def forward(self, x: Tensor, row_type: int, B: int, L: int) -> Tensor:
        """
        x: (B*L, K, D)
        row_type: int
        B: int
        L: int
        returns: (B, L, M)
        """
        field_embs = x.flatten(
            1
        )  # (B*L, K*D) concats all per-field embeddings into one vector per row
        projected = self.row_projs[row_type](field_embs)  # (B*L, M)
        return projected.view(B, L, self.M)  # (B, L, M)


# -------------------------------------------------------------------------------------- #
#  Row expander                                                                         #
# -------------------------------------------------------------------------------------- #
class RowExpander(nn.Module):
    """
    Expand row-level embeddings back to per-field embeddings.
    For row type h, applies a linear S_h: R^{M_row} -> R^{K_h * D_field},
    then reshapes to (B, L, K_h, D_field) for per-field heads.
    """

    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        K = len(schema.cat_features) + len(schema.cont_features)
        d_field = config.field_transformer.d_model
        d_row = config.sequence_transformer.d_model
        row_types = config.row_types
        self.K = K
        self.d_field = d_field

        # One expander per row type: S_h : M_row -> K_h * D_field
        self.expand = nn.ModuleList(
            [nn.Linear(d_row, K * d_field, bias=True) for _ in range(row_types)]
        )

    def forward(self, z_row: torch.Tensor, row_type: int = 0) -> torch.Tensor:
        """
        z_row: (B, L, M_row) - always expect full sequence
        returns: (B, L, K, D_field) - always return full sequence
        """
        B, L, _ = z_row.shape
        out = self.expand[row_type](z_row)  # (B, L, K*D)
        return out.view(B, L, self.K, self.d_field)  # (B, L, K, D)
