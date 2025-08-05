import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.embedder import TransformerEmbedder
from transaction_transformer.modeling.models.components.heads import FeaturePredictionHead


# -------------------------------------------------------------------------------------- #
#  Feature prediction model                                                                 #
# -------------------------------------------------------------------------------------- #
class FeaturePredictionModel(nn.Module):
    """
    Feature prediction model. Takes in a (B, L, M) embedding and returns logits for each feature.
    Can be used for next transaction prediction with the auto-regressive transformer, or masked token
    prediction with the masked transformer. Controlled by the is_causal flag in the config.

    Returns: A dictionary of length K = C + F, where each key is a feature name and each value is a (B, L, V_field) tensor.
    C is the number of categorical features, F is the number of continuous features.
    
    """
    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.config = config
        self.transaction_embedding_model = TransformerEmbedder(config, schema)
        self.feature_prediction_head = FeaturePredictionHead(config, schema)
        # No longer needed since both AR and MLM use same format

    def forward(self, cat: LongTensor, cont: Tensor, row_type: int = 0):
        embeddings = self.transaction_embedding_model(cat, cont, row_type) # (B, L, M)
        # For both AR and MLM, return predictions at all positions
        # The loss function will handle masking appropriately
        return self.feature_prediction_head(embeddings) # (B, L, M) -> dict[name]: (B, L, V_field) 