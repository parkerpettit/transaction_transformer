"""
transaction_model.py
--------------------
Clean, internally-consistent re-implementation of the tabular-sequence model.

Shapes (all tensors use batch-first convention):
    cat          : (B, L, C)   categorical token ids
    cont         : (B, L, F)   continuous features
    field_out    : (B*L, K, D)
    row_repr     : (B, L, M)
    seq_out      : (B, L, M)
    hidden       : (B, H)      final sequence representation
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

from networkx import in_degree_centrality
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor

from configs.config import (
    ModelConfig,
    TransformerConfig,
    LSTMConfig,
    MLPConfig
)

# -------------------------------------------------------------------------------------- #
#  Embedding + field transformer                                                         #
# -------------------------------------------------------------------------------------- #
class EmbeddingLayer(nn.Module):
    """
    Embeds categorical fields and uses frequency-based embeddings for continuous fields.
    
    For continuous features, we use sin/cos frequency embeddings:
    γ(v) = (sin(2^0*π*v), cos(2^0*π*v), ..., sin(2^(L-1)*π*v), cos(2^(L-1)*π*v))
    where L is the number of frequency pairs.

    Parameters
    ----------
    cat_vocab : Dict[str, int]
        Mapping field-name -> vocabulary size.
        **Order matters**: the columns in the `cat` tensor must follow this order.
    cont_feats : List[str]
        Names of continuous features.  Their order must match the columns in `cont`.
    emb_dim : int
        Dimension D of every field embedding / projection.
    dropout : float
        Dropout applied to each field embedding / projection.
    padding_idx : int
        Index used for PAD tokens in every categorical embedding.
    freq_pairs : int
        Number of sin/cos pairs for continuous feature embeddings (L=8 by default).

    Input shapes
    ------------
    cat  : (B, L, C)   integer ids for C categorical fields
    cont : (B, L, F)   floats for F continuous fields

    Output
    ------
    (B*L, K, D) where K = C + F
    """

    def __init__(
        self,
        cat_vocab: Dict[str, int],
        cont_feats: List[str],
        emb_dim: int,
        dropout: float,
        padding_idx: int,
        freq_pairs: int = 8,
    ) -> None:
        super().__init__()

        # Categorical embeddings (ModuleList keeps order)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(v, emb_dim, padding_idx=padding_idx) for v in cat_vocab.values()]
        )

        # Frequency-based continuous embeddings
        # Each continuous feature gets sin/cos pairs -> linear projection to emb_dim
        self.freq_pairs = freq_pairs
        freq_dim = 2 * freq_pairs  # sin/cos pairs
        self.cont_lin = nn.ModuleList(
            [nn.Linear(freq_dim, emb_dim) for _ in cont_feats]
        )
        
        # Pre-compute frequency multipliers: [2^0, 2^1, ..., 2^(L-1)]
        self.register_buffer('freq_multipliers', torch.pow(2.0, torch.arange(freq_pairs, dtype=torch.float32)))

        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.emb_dim = emb_dim
        self.num_fields = len(cat_vocab) + len(cont_feats)

    # --------------------------------------------------------------------- #
    def forward(self, cat: LongTensor, cont: Tensor) -> Tensor:
        """
        Returns
        -------
        field_repr : Tensor
            Shape (B*L, K, D) where K = #categorical + #continuous fields.
        """
        B, L, _ = cat.shape

        # categorical → (B, L, D) for each field
        cat_embs = [
            self.dropout(layer(cat[:, :, i]))
            for i, layer in enumerate(self.cat_emb)
        ]

        # continuous → frequency-based embeddings → (B, L, D) for each field
        cont_embs = []
        for i, layer in enumerate(self.cont_lin):
            # Get continuous values for this feature: (B, L)
            values = cont[:, :, i]  # (B, L)
            
            # Compute frequency embeddings: γ(v) = (sin(2^0*π*v), cos(2^0*π*v), ...)
            freq_values = values.unsqueeze(-1) * self.freq_multipliers.unsqueeze(0).unsqueeze(0)  # (B, L, freq_pairs)
            freq_values = freq_values * math.pi  # Scale by π
            
            # Create sin/cos pairs
            sin_vals = torch.sin(freq_values)  # (B, L, freq_pairs)
            cos_vals = torch.cos(freq_values)  # (B, L, freq_pairs)
            
            # Interleave sin/cos: [sin(2^0*π*v), cos(2^0*π*v), sin(2^1*π*v), cos(2^1*π*v), ...]
            freq_embed = torch.stack([sin_vals, cos_vals], dim=-1)  # (B, L, freq_pairs, 2)
            freq_embed = freq_embed.reshape(B, L, -1)  # (B, L, 2*freq_pairs)
            
            # Project to embedding dimension
            cont_emb = self.dropout(layer(freq_embed))  # (B, L, D)
            cont_embs.append(cont_emb)

        # stack along field dimension K
        field_stack = torch.stack(cat_embs + cont_embs, dim=2)  # (B, L, K, D)
        return field_stack.reshape(B * L, self.num_fields, self.emb_dim)

# field_transformer.py
import torch
import torch.nn as nn
from torch import Tensor
from configs.config import TransformerConfig


class FieldTransformer(nn.Module):
    """
    Intra-row Transformer that safely handles very large (B*L) row batches.

    * Input / output shape:  (B x L, K, D)  (batch_first = True)
    * Automatically splits the batch dimension into ≤ 65 535-row chunks so the
      underlying CUDA kernel never exceeds the grid-dim.x limit.
    * Uses views, so the split / cat adds almost zero overhead.

    Parameters
    ----------
    cfg : TransformerConfig
        Hyper-parameters for the encoder layer (d_model, n_heads, dropout, …)
    """

    _MAX_ROWS = 60_000          # CUDA gridDim.x hard limit

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.ffn_mult * cfg.d_model,
            dropout         = cfg.dropout,
            activation      = "relu",
            layer_norm_eps  = cfg.layer_norm_eps,
            batch_first     = True,          # (B*L, K, D)
            norm_first      = cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=cfg.depth)

    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:         # x: (B*L, K, D)
        """
        Returns
        -------
        Tensor
            Same shape as `x`, with intra-row attention applied.
        """
        n_rows = x.size(0)

        # Fast path - fits in one kernel launch
        if n_rows <= self._MAX_ROWS:
            return self.encoder(x)

        # Chunked path - split along the row dimension
        out_chunks = []
        for chunk in x.split(self._MAX_ROWS, dim=0):   # view; zero-copy
            out_chunks.append(self.encoder(chunk))
        return torch.cat(out_chunks, dim=0)


# -------------------------------------------------------------------------------------- #
#  Sequence transformer (temporal)                                                      #
# -------------------------------------------------------------------------------------- #
class SinCosPositionalEncoding(nn.Module):
    """
    "Attention Is All You Need” sinusoidal positional encoding.
    Works with batch-first tensors (B, L, D).

    Parameters
    ----------
    d_model : int
        Embedding dimension (must match the model's hidden size).
    max_len : int, default = 2048
        Maximum sequence length you expect to handle.  A single
        (max_len x d_model) matrix is pre-computed and stored as a
        non-persistent buffer; slices are taken at runtime.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Pre-compute the full positional encoding matrix.
        position   = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)        # (max_len, 1)
        div_term   = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                               (-math.log(10000.0) / d_model))                      # (d_model/2,)
        pe         = torch.zeros(max_len, d_model, dtype=torch.float32)             # (max_len, D)
        pe[:, 0::2] = torch.sin(position * div_term)                                # even indices
        pe[:, 1::2] = torch.cos(position * div_term)                                # odd  indices
        pe         = pe.unsqueeze(0)                                                # (1, max_len, D)

        # Register as a buffer so it moves with .to(device) but isn’t saved in checkpoints.
        self.register_buffer("pe", pe, persistent=False)

    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to `x`.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, L, D) with `D == self.d_model`.

        Returns
        -------
        Tensor
            Same shape as `x`, with positional encodings added.
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        L = x.size(1)
        if L > self.pe.size(1): # type: ignore
            raise ValueError(f"Sequence length {L} exceeds max_len={self.pe.size(1)}") # type: ignore

        # self.pe is (1, max_len, D) - slice to length and broadcast over batch
        return x + self.pe[:, :L] # type: ignore


class SequenceTransformer(nn.Module):
    """Causal Transformer encoder over the temporal dimension."""

    def __init__(self, cfg: TransformerConfig, max_len: int = 256):
        super().__init__()
        self.pos_enc = SinCosPositionalEncoding(cfg.d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="relu",
            layer_norm_eps=cfg.layer_norm_eps,
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()  # (max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)


    def forward(self, x: Tensor) -> Tensor:  # (B, L, M)
        L = x.size(1)
        causal = self.causal_mask[:L, :L] # type: ignore
        x = self.pos_enc(x)
        return self.encoder(x, mask=causal)


# -------------------------------------------------------------------------------------- #
#  LSTM head                                                                            #
# -------------------------------------------------------------------------------------- #
class LSTMHead(nn.Module):
    """
    LSTM-based sequence summarization + classification head.
    Takes in a (B, L, M) sequence, returns (B, num_classes) logits.


    """

    def __init__(self, cfg: LSTMConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout
        )
        self.fc = nn.Linear(cfg.hidden_size, 1)

    def forward(self, seq_out: Tensor) -> Tensor:
        """
        Parameters
        ----------
        seq_out : Tensor
            Shape (B, L, M) - output from the sequence transformer.
        lengths : Tensor
            Shape (B,)

        Returns
        -------
        logits : Tensor
            Shape (B,) - unnormalized fraud classification logits
        """

        embeddings, _ = self.lstm(seq_out)
        last_embedding = embeddings[:, -1, :] # (B, hidden_size)
        return self.fc(last_embedding).squeeze(dim=1)  # (B)


class FraudHeadMLP(nn.Module):
    def __init__(self, emb_dim: int, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        # If only one layer, do a plain linear
        if cfg.num_layers == 1:
            layers.append(nn.Linear(emb_dim, 1))

        # Otherwise build: Input → hidden → … → hidden → output
        else:
            # first hidden layer
            layers.append(nn.Linear(emb_dim, cfg.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.dropout))

            # middle hidden layers (if num_layers > 2)
            for _ in range(cfg.num_layers - 2):
                layers.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(cfg.dropout))

            # final output layer
            layers.append(nn.Linear(cfg.hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, emb_dim)
        return self.net(x).squeeze(dim=1) #(B,)
# -------------------------------------------------------------------------------------- #
#  Full model                                                                           #
# -------------------------------------------------------------------------------------- #
class TransactionModel(nn.Module):
    """Tabular-sequence model with optional LSTM fraud-classification head."""

    # --------------------------------------------------------------------- #
    def __init__(self, cfg: ModelConfig):
        """
        Parameters
        ----------
        cfg      : ModelConfig
        max_len  : override for maximum causal-mask length; if None uses cfg.window
        """
        super().__init__()
        self.cfg: ModelConfig = cfg

        # -- Embedding & field transformer ----------------------------------
        self.embedder = EmbeddingLayer(
            cat_vocab=cfg.cat_vocab_sizes,
            cont_feats=cfg.cont_features,
            emb_dim=cfg.ft_config.d_model,       # emb_dim == ft.d_model (validated in cfg)
            dropout=cfg.emb_dropout,
            padding_idx=cfg.padding_idx,
            freq_pairs=8,  # Number of sin/cos pairs for frequency embeddings
        )
        self.field_tf = FieldTransformer(cfg.ft_config)

        # -- Row projection (flatten K*D → M) -------------------------------
        self.num_fields = len(cfg.cat_vocab_sizes) + len(cfg.cont_features)
        self.row_proj   = nn.Linear(self.num_fields * cfg.ft_config.d_model,
                                    cfg.seq_config.d_model)

        # -- Sequence transformer -------------------------------------------
        self.seq_tf = SequenceTransformer(cfg.seq_config, max_len=cfg.window)

        # -- Optional LSTM fraud head ---------------------------------------
        self.lstm_head: nn.Module | None = None
        if cfg.lstm_config is not None:
            self.add_lstm_head(cfg.lstm_config)

        self.mlp_head: nn.Module | None = None
        if cfg.mlp_config is not None:
            self.add_mlp_head(cfg.mlp_config)

        # -- Autoregressive head (predict next transaction) - unified for all features
        self.ar_dropout = nn.Dropout(cfg.clf_dropout)
        # Total vocabulary: categorical + quantized continuous
        total_vocab_size = sum(cfg.cat_vocab_sizes.values())
        if cfg.use_quantized_targets and cfg.cont_vocab_sizes:
            total_vocab_size += sum(cfg.cont_vocab_sizes.values())
        else:
            # Fallback: treat continuous as single regression target per feature
            total_vocab_size += len(cfg.cont_features)
        self.ar_head = nn.Linear(cfg.seq_config.d_model, total_vocab_size)

        # -- MLM head (predict masked tokens) - unified for all features
        self.mlm_dropout = nn.Dropout(cfg.clf_dropout)
        self.mlm_head = nn.Linear(cfg.seq_config.d_model, total_vocab_size)

      


    def add_lstm_head(self, lstm_cfg: LSTMConfig) -> None:
        self.cfg.lstm_config = lstm_cfg
        self.lstm_head = LSTMHead(lstm_cfg, input_size=self.cfg.seq_config.d_model)

    def add_mlp_head(self, mlp_cfg: MLPConfig) -> None:
        self.cfg.mlp_config = mlp_cfg
        self.mlp_head = FraudHeadMLP(
            emb_dim=self.cfg.seq_config.d_model,
            cfg  = mlp_cfg,
        )




    # --------------------------------------------------------------------- #
    def forward(
        self,
        cat: LongTensor,        # (B, L, C)
        cont: Tensor,           # (B, L, F)
        mode: str = "ar",
        mask: Optional[torch.Tensor] = None,  # (B, L) for efficient MLM
    ):
        # -- 1) field-level attention -----------------------------------------
        field = self.embedder(cat, cont)          # (B*L, K, D)
        field = self.field_tf(field)              # (B*L, K, D)  (chunked if needed)
        field = field.flatten(1)                  # (B*L, K*D) concats all per-field embeddings into one vector per row

        # -- 2) project row, then temporal transformer -----------------------
        B, L, _ = cat.shape
        row = self.row_proj(field).view(B, L, -1) # (B, L, M)
        seq = self.seq_tf(row)          # (B, L, M)

        # -- 4) heads --------------------------------------------------------
         # -- heads ----------------------------------------------------------
        if mode == "lstm":
            if self.lstm_head is None:
                raise RuntimeError("No LSTM head attached.")
            return self.lstm_head(seq)

        if mode == "mlp":
            if self.mlp_head is None:
                raise RuntimeError("No MLP head attached.")
            return self.mlp_head(seq[:, -1, :])

        if mode == "ar":
            h = seq[:, -1, :]
            z = self.ar_dropout(h)
            return self.ar_head(z)  # (B, total_vocab_size)

        if mode == "masked" or mode == "mlm":
            # For MLM, we can optionally compute only masked positions for efficiency
            if mask is not None and mask.sum() > 0:
                # Check if mask is 3D (field-level) or 2D (row-level)
                if len(mask.shape) == 3:
                    # 3D field-level mask: compute full sequence logits
                    z = self.mlm_dropout(seq)  # (B, L, D)
                    logits = self.mlm_head(z)  # (B, L, total_vocab_size)
                    return logits
                else:
                    # 2D row-level mask: use efficient sparse computation
                    masked_positions = mask.nonzero(as_tuple=False)  # (num_masked, 2) -> [batch_idx, seq_idx]
                    if len(masked_positions) > 0:
                        # Extract representations only for masked positions
                        batch_indices = masked_positions[:, 0]  # (num_masked,)
                        seq_indices = masked_positions[:, 1]    # (num_masked,)
                        masked_repr = seq[batch_indices, seq_indices, :]  # (num_masked, D)
                        
                        # Compute logits only for masked positions
                        z = self.mlm_dropout(masked_repr)  # (num_masked, D)
                        logits = self.mlm_head(z)  # (num_masked, total_vocab_size)
                        
                        return logits, masked_positions  # Return logits + position info
                    else:
                        # No masked positions
                        out_features = self.ar_head.out_features if hasattr(self.ar_head, 'out_features') else self.mlm_head.out_features
                        return torch.empty(0, out_features, device=seq.device), torch.empty(0, 2, dtype=torch.long, device=seq.device)
            else:
                # Fallback: compute full sequence (original behavior)
                z = self.mlm_dropout(seq)  # (B, L, D)
                logits = self.mlm_head(z)  # (B, L, total_vocab_size)
                return logits

        if mode == "extract":
            return seq[:, -1, :]

        raise ValueError("mode must be 'ar', 'lstm', 'mlp', 'masked', or 'mlm'")