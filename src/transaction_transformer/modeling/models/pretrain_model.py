import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.backbone import Backbone
from transaction_transformer.modeling.models.components.heads import PretrainHead


# -------------------------------------------------------------------------------------- #
#  Pretraining model                                                                     #
# -------------------------------------------------------------------------------------- #
class PretrainingModel(nn.Module):
    """
    Pretraining model. Takes in (B, L, M) embeddings from the backbone and returns
    logits for each feature via the pretraining head.

    Returns: A dictionary of length K = C + F, where each key is a feature name and each value is a (B, L, V_field) tensor.
    C is the number of categorical features, F is the number of continuous features.
    """

    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.config = config
        self.backbone = Backbone(config, schema)
        self.head = PretrainHead(config, schema)

    def forward(self, cat: LongTensor, cont: Tensor, row_type: int = 0):
        embeddings = self.backbone(cat, cont, row_type)  # (B, L, M)
        return self.head(embeddings)  # dict[name]: (B, L, V_field)
