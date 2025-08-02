import torch.nn as nn
from torch import Tensor, LongTensor
from transaction_transformer.config.config import ModelConfig
from transaction_transformer.modeling.models.embedder import TransformerEmbedder
from transaction_transformer.modeling.models.components import ClassificationHead
from transaction_transformer.modeling.models.predictor import FeaturePredictionModel
from transaction_transformer.data.preprocessing.tokenizer import FieldSchema

# -------------------------------------------------------------------------------------- #
#  Fraud detection model                                                                 #
# -------------------------------------------------------------------------------------- #
class FraudDetectionModel(nn.Module):
    """
    Fraud detection model. Takes a pretrained transaction embedding model and a fraud classification head.
    Returns a (B,) logits. Can be set to finetune end-to-end or train the fraud classification head alone
    on top of the frozen transaction embedding model. Controlled by the freeze_embedding flag in the config.
    Output is log-probabilities.
    """
    #TODO come back and make this work. idea is to have a pretrained transaction embedding model saved,
    # and then load it here and attach a fraud classification head to it. can freeze the embedding model
    # and train the fraud classification head alone, or finetune the entire model end-to-end.

    def __init__(self, config: ModelConfig, schema: FieldSchema):
        super().__init__()
        self.config = config
        self.transaction_embedding_model = TransformerEmbedder(config, schema)
        self.transaction_embedding_model.requires_grad_(not config.freeze_embedding)
        self.classification_head = ClassificationHead(config)
        self.feature_prediction_model = FeaturePredictionModel(config, schema)
        self.feature_prediction_model.requires_grad_(not config.freeze_embedding)
       
    def forward(self, cat: LongTensor, cont: Tensor, row_type: int = 0):
        embeddings = self.transaction_embedding_model(cat, cont, row_type)
        return self.classification_head(embeddings)
       