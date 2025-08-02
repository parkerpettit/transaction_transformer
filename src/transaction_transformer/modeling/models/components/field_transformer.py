import torch.nn as nn
from torch import Tensor
from transaction_transformer.config.config import TransformerConfig

class FieldTransformer(nn.Module):
    """
    Intra-row Transformer.

    * Input / output shape:  (B*L, K, D)  (batch_first = True)

    Parameters
    ----------
    config : TransformerConfig
        Hyper-parameters for the encoder layer (d_model, n_heads, dropout, …)
    """


    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.d_model,
            nhead           = config.n_heads,
            dim_feedforward = config.ffn_mult * config.d_model,
            dropout         = config.dropout,
            activation      = "gelu",
            layer_norm_eps  = config.layer_norm_eps,
            batch_first     = True,          
            norm_first      = config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=config.depth)

    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:         # x: (B*L, K, D)
        """
        Returns
        -------
        Tensor
            Same shape as `x`, with intra-row attention applied.
        """
        return self.encoder(x) # (B*L, K, D) 
