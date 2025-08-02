import torch
import torch.nn as nn
from torch import Tensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.components.projection import RowExpander
from transaction_transformer.utils import build_mlp
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema

# -------------------------------------------------------------------------------------- #
#  Feature prediction head                                                                 #
# -------------------------------------------------------------------------------------- #
class FeaturePredictionHead(nn.Module):
    """
    Attribute-specific heads. Takes (B, L, M) embeddings and returns logits per field: dict[field_name] -> (B, L, V_f).
    """
    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.row_expander = RowExpander(config, schema)
        m = config.sequence_transformer.d_model
        cat_names = schema.cat_features
        cont_names = schema.cont_features
        all_names = cat_names + cont_names

        heads = nn.ModuleDict()
        d_field = config.field_transformer.d_model
        for name in cat_names:
            V = schema.cat_encoders[name].vocab_size
            heads[name] = build_mlp(
                input_dim=d_field,
                hidden_dim=config.classification.hidden_dim,
                output_dim=V,
                depth=config.classification.depth,
                dropout=config.classification.dropout,
            )
        for name in cont_names:
            V = schema.cont_binners[name].num_bins
            heads[name] = build_mlp(
                input_dim=d_field,
                hidden_dim=config.classification.hidden_dim,
                output_dim=V,
                depth=config.classification.depth,
                dropout=config.classification.dropout,
            )
        self.heads = heads
        self.all_names = all_names

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, L, M) input from the sequence transformer
        returns: dict[name] -> (B, L, V_field) which are the logits for each feature at each time step, 
                where V_field is the vocabulary size for that feature. Dict is of length K = C + F.
                First C features are categorical, last F features are continuous.
       """
        logits: dict[str, torch.Tensor] = {}
        z_row = self.row_expander(x) # (B, L, M) -> (B, L, K, D_field)
        for k, name in enumerate(self.all_names):
            logits[name] = self.heads[name](z_row[:, :, k, :]) # (B, L, D_field) -> (B, L, V_field)
        return logits # dict[name]: (B, L, V_field)

# -------------------------------------------------------------------------------------- #
#  Classification head                                                             #
# -------------------------------------------------------------------------------------- #
class ClassificationHead(nn.Module):
    """
    Classification head. Takes in a (B, L, M) embedding and returns a (B,) logits for the fraud classification.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = build_mlp(
            input_dim=config.sequence_transformer.d_model,
            hidden_dim=config.classification.hidden_dim, 
            output_dim=config.classification.output_dim,
            depth=config.classification.depth,
            dropout=config.classification.dropout 
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) # (B, L, M) -> (B, L) -> (B,)