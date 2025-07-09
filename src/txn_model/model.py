import torch
import torch.nn as nn
from torch import Tensor, LongTensor

class EmbeddingLayer(nn.Module):
    """
    Embedding layer for tabular transactions.

    - Embeds each of C categorical fields via individual nn.Embedding tables
    - Projects F continuous fields via individual nn.Linear layers
    - Returns a (B*L, K, D) tensor where K is the total number of features (C + F).

    Args:
        cat_vocab_sizes: mapping from cat_feature -> vocab size
        cont_features: list of continuous feature names (length F)
        emb_dim: embedding dimension D (for both cat + cont)
        dropout: dropout probability applied after each embedding lookup
        padding_idx: index in each categorical embedding to treat as padding
    """
    def __init__(self,
                    cat_vocab_sizes: dict[str, int],
                    cont_features: list[str],
                    emb_dim: int,
                    dropout: float,
                    padding_idx: int):
            super().__init__()
            self.cat_features = list(cat_vocab_sizes)
            self.cont_features = cont_features
            # categorical embeddings
            self.cat_embeds = nn.ModuleDict({
                f: nn.Embedding(cat_vocab_sizes[f], emb_dim, padding_idx)
                for f in self.cat_features
            })
            # numeric: one projection per field
            self.cont_proj = nn.ModuleDict({
                f: nn.Linear(1, emb_dim) for f in self.cont_features
            })
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, cat: LongTensor, cont: Tensor) -> Tensor:
        B, L, _ = cat.shape
        D = next(iter(self.cat_embeds.values())).embedding_dim

        cat_embs = [self.dropout(self.cat_embeds[f](cat[:, :, i]))
                    for i, f in enumerate(self.cat_features)]
        cont_embs = [self.dropout(self.cont_proj[f](cont[:, :, i].unsqueeze(-1)))
                     for i, f in enumerate(self.cont_features)]

        tokens = torch.stack(cat_embs + cont_embs, dim=2)   # (B,L,K,D)
        return tokens.reshape(B * L, -1, D)                 # (B*L,K,D)


import torch
import torch.nn as nn
from torch import Tensor
class FieldTransformer(nn.Module):
    """
    Transformer for capturing intra-row interactions.

    Input:
          x:  Tensor of shape (N, K, D) where
              N = B × L  rows in the mini-batch
              K = C + F  #fields in one row
              D = d_model embedding width

    Output:
          Tensor of the same dimensions
    """
    def __init__(
        self,
        d_model: int, # embedding dimension size
        n_heads: int,
        depth: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,

    ):
      super().__init__()
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=n_heads,
          dim_feedforward=ffn_mult*d_model,
          dropout=dropout,
          activation="relu",
          layer_norm_eps=layer_norm_eps,
          batch_first=True,
          norm_first=norm_first,
          bias=True
      )
      self.encoder = nn.TransformerEncoder(
          encoder_layer=encoder_layer,
          num_layers=depth,
          norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
          enable_nested_tensor=True
      )

    def forward(self, x: Tensor) -> Tensor:
      """
      Input:
      x:  Tensor of shape (N, K, D) where
          N = B × L  rows in the mini-batch
          K = C + F  #fields in one row
          D = d_model embedding width

      Output:
          Tensor of the same dimensions after Transformer processing
      """
      return self.encoder(x)

import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    """
    Classic sin–cos positional encoding.
    Re-computed each call; no internal cache.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, L, D)
        returns x + PE  (same shape, same device)
        """
        B, L, D = x.shape
        device  = x.device

        # position indices 0 … L-1  →  (L, 1)
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)

        # 10000^(2i/D) term   (D_even entries)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / D)
        )                                                # (D/2,)

        pe = torch.zeros(L, D, device=device)            # (L, D)
        pe[:, 0::2] = torch.sin(pos * div_term)          # even dims
        pe[:, 1::2] = torch.cos(pos * div_term)          # odd  dims

        return x + pe.unsqueeze(0)                       # broadcast over batch


class SequenceTransformer(nn.Module):
  """
  Transformer for capturing inter-row interactions.

  Input:
        A tensor of shape (B, L, m) from the first LinearLayer, where
        B = batch size, L = sequence length, m = row embedding dimension

  Output:
        A tensor of shape (B, L, m) with time contextualization
  """
  def __init__(self,
        d_model: int, # m
        n_heads: int,
        depth: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
               ):
    super().__init__()
    self.pos_encoder = PositionalEncoding()

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_mult*d_model,
        dropout=dropout,
        activation="relu",
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=norm_first
    )

    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=depth,
    )

  def forward(self, x: Tensor, padding_mask):
    """
    Args:
        x: a tensor of shape (B, L, m)
    Returns:
        A tensor of the same shape with time contextualization
    """
    _, L, _ = x.shape
    x = self.pos_encoder(x)
    mask = nn.Transformer.generate_square_subsequent_mask(L, dtype=torch.bool, device=x.device)
    return self.transformer_encoder(x, src_key_padding_mask=padding_mask, is_causal=True, mask=mask)




import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
  """
  Final MLP for classification + regression. Takes in a (B*L, d*k) shape tensor
  from the SecondLinearLayer. Returns class logits and regression values.
  """
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      class_dims: list[int],  # e.g. [vocab_size_feat0, vocab_size_feat1, …]
      num_cont: int
  ):
      super().__init__()
      self.class_dims = class_dims
      self.num_cont   = num_cont
      total_class    = sum(class_dims)
      self.net = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_dim, total_class + num_cont),
      )

  def forward(self, x: Tensor):
      # x: (B*L, input_dim)
      out = self.net(x)  # → (B*L, total_class + num_cont)

      # split out the class logits
      class_logits = out[..., :sum(self.class_dims)]  # (B*L, total_class)
      regs         = out[..., sum(self.class_dims):]  # (B*L, num_cont)

      # now slice class_logits into a list, one per feature
      logits_list, offset = [], 0
      for sz in self.class_dims:
          logits_list.append(class_logits[..., offset:offset+sz])  # (B*L, sz)
          offset += sz

      return logits_list, regs



from torch import BoolTensor

class TransactionModel(nn.Module):
  """
  A full end to end model.
  """
  def __init__(self, config: ModelConfig):
    super().__init__()

    num_cat = len(config.cat_vocab_sizes)
    num_cont = len(config.cont_features)
    k = num_cat + num_cont
    d = config.emb_dim
    m = config.sequence_transformer.d_model

    self.embedder = EmbeddingLayer(
        cat_vocab_sizes=config.cat_vocab_sizes,
        cont_features=config.cont_features,
        emb_dim=config.emb_dim,
        dropout=config.dropout,
        padding_idx=config.padding_idx
    )

    ft_cfg = config.field_transformer
    self.field_transformer = FieldTransformer(
        d_model=ft_cfg.d_model,
        n_heads=ft_cfg.n_heads,
        depth=ft_cfg.depth,
        ffn_mult=ft_cfg.ffn_mult,
        dropout=ft_cfg.dropout,
        layer_norm_eps=ft_cfg.layer_norm_eps,
        norm_first=ft_cfg.norm_first
    )


    st_cfg = config.sequence_transformer
    self.sequence_transformer = SequenceTransformer(
        d_model=st_cfg.d_model,
        n_heads=st_cfg.n_heads,
        depth=st_cfg.depth,
        ffn_mult=st_cfg.ffn_mult,
        dropout=st_cfg.dropout,
        layer_norm_eps=st_cfg.layer_norm_eps,
        norm_first=st_cfg.norm_first
    )


    # self.head = MLP(
    #     input_dim=config.emb_dim * k,
    #     hidden_dim=config.mlp_hidden,
    #     class_dims=list(config.cat_vocab_sizes.values()),
    #     num_cont=num_cont
    # ) # change to lstm head

  def forward(self, cat: LongTensor, cont: Tensor, padding_mask: BoolTensor):
    B, L, _ = cat.shape

    embeddings = self.embedder(cat, cont)
    intra_row_out = self.field_transformer(embeddings).flatten(start_dim=1)
    row_repr = self.row_projector(intra_row_out).view(B, L, -1)
    seq_out = self.sequence_transformer(row_repr, padding_mask=padding_mask)
    final_repr = self.output_projector(seq_out.view(B*L, -1))

    class_logits, regression_vals = self.head(final_repr)
    class_logits = [
          l.view(B, L, -1) for l in class_logits
    ]
    regression_vals = regression_vals.view(B, L, -1)

    return class_logits, regression_vals










