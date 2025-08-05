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
    Attribute-specific heads. Takes (B, M) embeddings and returns logits per field: dict[field_name] -> (B, V_f).
    """
    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.row_expander = RowExpander(config, schema)
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
        self.is_causal = config.training.model_type == "ar"

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, L, M) input from the sequence transformer if mlm, (B, M) if ar
        returns: dict[name] -> (B, L, V_field) which are the logits for each feature if mlm,
                (B, V_field) if ar, where V_field is the vocabulary size for that feature.
                Dict is of length K = C + F. L is the number of rows in the sequence.
       """
        logits: dict[str, torch.Tensor] = {}
        z_row = self.row_expander(x) # (B, L, M) -> (B, L, K, D_field) if mlm, (B, K, D_field) if ar
        for k, name in enumerate(self.all_names):
            if self.is_causal: # ar model_type
                logits[name] = self.heads[name](z_row[:, k, :]) # (B, K, D_field) -> (B, V_field)
            else: # mlm model_type
                logits[name] = self.heads[name](z_row[:, :, k, :]) # (B, L, K, D_field) -> (B, L, V_field)
        return logits # dict[name]: (B, L, V_field) if mlm, (B, V_field) if ar

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