"""
transaction_model.py
--------------------
Clean, internally-consistent re-implementation of the tabular-sequence model.

Shapes (all tensors use batch-first convention):
    cat          : (B, L, C)   categorical token ids
    cont         : (B, L, F)   continuous features
    pad_mask     : (B, L)      True for PAD positions
    field_out    : (B·L, K, D)
    row_repr     : (B, L, M)
    seq_out      : (B, L, M)
    hidden       : (B, H)      final sequence representation
"""

from __future__ import annotations
import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (
    ModelConfig,
    TransformerConfig,
    LSTMConfig,
)

# -------------------------------------------------------------------------------------- #
#  Embedding + field transformer                                                         #
# -------------------------------------------------------------------------------------- #
class EmbeddingLayer(nn.Module):
    """
    Embeds categorical fields and linearly projects continuous fields.

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

    Input shapes
    ------------
    cat  : (B, L, C)   integer ids for C categorical fields
    cont : (B, L, F)   floats for F continuous fields

    Output
    ------
    (B·L, K, D) where K = C + F
    """

    def __init__(
        self,
        cat_vocab: Dict[str, int],
        cont_feats: List[str],
        emb_dim: int,
        dropout: float,
        padding_idx: int,
    ) -> None:
        super().__init__()

        # Categorical embeddings (ModuleList keeps order)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(v, emb_dim, padding_idx=padding_idx) for v in cat_vocab.values()]
        )

        # Continuous projections 1→D (bias not needed; remove if you do want bias)
        self.cont_lin = nn.ModuleList(
            [nn.Linear(1, emb_dim, bias=False) for _ in cont_feats]
        )

        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.emb_dim = emb_dim
        self.num_fields = len(cat_vocab) + len(cont_feats)

    # --------------------------------------------------------------------- #
    def forward(self, cat: LongTensor, cont: Tensor) -> Tensor:
        """
        Returns
        -------
        field_repr : Tensor
            Shape (B·L, K, D) where K = #categorical + #continuous fields.
        """
        B, L, _ = cat.shape

        # categorical → (B, L, D) for each field
        cat_embs = [
            self.dropout(layer(cat[:, :, i]))
            for i, layer in enumerate(self.cat_emb)
        ]

        # continuous → (B, L, D) for each field
        cont_embs = [
            self.dropout(layer(cont[:, :, i].unsqueeze(-1)))
            for i, layer in enumerate(self.cont_lin)
        ]

        # stack along field dimension K
        field_stack = torch.stack(cat_embs + cont_embs, dim=2)  # (B, L, K, D)
        return field_stack.reshape(B * L, self.num_fields, self.emb_dim)

# field_transformer.py
import torch
import torch.nn as nn
from torch import Tensor
from config import TransformerConfig


class FieldTransformer(nn.Module):
    """
    Intra-row Transformer that safely handles very large (B·L) row batches.

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
    def forward(self, x: Tensor) -> Tensor:         # x: (B·L, K, D)
        """
        Returns
        -------
        Tensor
            Same shape as `x`, with intra-row attention applied.
        """
        n_rows = x.size(0)

        # Fast path – fits in one kernel launch
        if n_rows <= self._MAX_ROWS:
            return self.encoder(x)

        # Chunked path – split along the row dimension
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

        # self.pe is (1, max_len, D) – slice to length and broadcast over batch
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


    def forward(self, x: Tensor, pad_mask: BoolTensor) -> Tensor:  # (B, L, M)
        L = x.size(1)
        causal = self.causal_mask[:L, :L] # type: ignore
        x = self.pos_enc(x)
        return self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)


# -------------------------------------------------------------------------------------- #
#  LSTM head                                                                            #
# -------------------------------------------------------------------------------------- #
class LSTMHead(nn.Module):
    """
    LSTM-based sequence summarization + classification head.
    Takes in a (B, L, M) sequence, returns (B, num_classes) logits.

    If all input sequences are padded to the same length, packing is skipped.
    """

    def __init__(self, cfg: LSTMConfig, input_size: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout
        )

        self.dropout = nn.Dropout(cfg.dropout)

        self.fc = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, seq_out: Tensor, lengths: Tensor) -> Tensor:
        """
        Parameters
        ----------
        seq_out : Tensor
            Shape (B, L, M) - output from the sequence transformer.
        lengths : Tensor
            Shape (B,) - number of unpadded tokens per sequence (i.e., (~pad_mask).sum(1))

        Returns
        -------
        logits : Tensor
            Shape (B, num_classes) - unnormalized fraud classification logits
        """

        # If all sequences are same length, we can skip packing for efficiency
        if torch.all(lengths == lengths[0]):
            _, (h_n, _) = self.lstm(seq_out)
        else:
            packed = pack_padded_sequence(seq_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)

        last_hidden = h_n[-1]  # (B, hidden_size)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)  # (B, num_classes)


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
        self.cfg = cfg

        # ── Embedding & field transformer ──────────────────────────────────
        self.embedder = EmbeddingLayer(
            cat_vocab=cfg.cat_vocab_sizes,
            cont_feats=cfg.cont_features,
            emb_dim=cfg.ft_config.d_model,       # emb_dim == ft.d_model (validated in cfg)
            dropout=cfg.emb_dropout,
            padding_idx=cfg.padding_idx,
        )
        self.field_tf = FieldTransformer(cfg.ft_config)

        # ── Row projection (flatten K·D → M) ───────────────────────────────
        self.num_fields = len(cfg.cat_vocab_sizes) + len(cfg.cont_features)
        self.row_proj   = nn.Linear(self.num_fields * cfg.ft_config.d_model,
                                    cfg.seq_config.d_model)

        # ── Sequence transformer ───────────────────────────────────────────
        self.seq_tf = SequenceTransformer(cfg.seq_config, max_len=cfg.window)

        # ── Optional LSTM fraud head ───────────────────────────────────────
        if self.cfg.lstm_config is not None:
            self.lstm_head = LSTMHead(self.cfg.lstm_config, input_size=cfg.seq_config.d_model)
        else:
            self.lstm_head = None
        # ── Autoregressive heads (always present) ──────────────────────────
        
        self.ar_dropout   = nn.Dropout(cfg.clf_dropout)
        self.ar_cat_head  = nn.Linear(cfg.seq_config.d_model,
                                      sum(cfg.cat_vocab_sizes.values()))
        self.ar_cont_head = nn.Linear(cfg.seq_config.d_model, len(cfg.cont_features))

    # --------------------------------------------------------------------- #
    def forward(
        self,
        cat: LongTensor,        # (B, L, C)
        cont: Tensor,           # (B, L, F)
        pad_mask: BoolTensor,   # (B, L)  True → PAD
        mode: str = "fraud",    # 'fraud' | 'ar'
    ):
        # ── 1) field-level attention ─────────────────────────────────────────
        field = self.embedder(cat, cont)          # (B·L, K, D)
        field = self.field_tf(field)              # (B·L, K, D)  (chunked if needed)
        field = field.flatten(1)                  # (B·L, K·D) concats all per-field embeddings into one vector per row

        # ── 2) project row, then temporal transformer ───────────────────────
        B, L, _ = cat.shape
        row = self.row_proj(field).view(B, L, -1) # (B, L, M)
        seq = self.seq_tf(row, pad_mask)          # (B, L, M)

        # ── 3) sequence lengths (keep on same device) ───────────────────────
        lengths = (~pad_mask).sum(dim=1)          # (B,)  stays on GPU

        # ── 4) heads ────────────────────────────────────────────────────────
        if mode == "fraud":
            if self.lstm_head is None:
                raise RuntimeError("Fraud head not present (pre-train config).")
            return self.lstm_head(seq, lengths)   # (B, 1)

        elif mode == "ar":
            idx = lengths - 1                     # last valid time-step per batch
            hidden = seq[torch.arange(B, device=seq.device), idx]  # (B, M or H)
            z = self.ar_dropout(hidden)
            return self.ar_cat_head(z), self.ar_cont_head(z)

        else:
            raise ValueError("mode must be 'fraud' or 'ar'")
