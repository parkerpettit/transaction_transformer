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
    FieldTransformerConfig,
    SequenceTransformerConfig,
    LSTMConfig,
)

# -------------------------------------------------------------------------------------- #
#  Embedding + field transformer                                                         #
# -------------------------------------------------------------------------------------- #
class EmbeddingLayer(nn.Module):
    """Embeds C categoricals and linearly projects F continuous features."""

    def __init__(
        self,
        cat_vocab: Dict[str, int],
        cont_features: List[str],
        emb_dim: int,
        dropout: float,
        padding_idx: int,
    ):
        super().__init__()
        self.cat_keys = list(cat_vocab)          # preserve ordering
        self.cont_keys = cont_features

        self.cat_emb = nn.ModuleDict(
            {k: nn.Embedding(v, emb_dim, padding_idx) for k, v in cat_vocab.items()}
        )
        self.cont_lin = nn.ModuleDict(
            {k: nn.Linear(1, emb_dim) for k in cont_features}
        )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, cat: LongTensor, cont: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor (B·L, K, D) where K = C + F
        """
        B, L, _ = cat.shape
        D = next(iter(self.cat_emb.values())).embedding_dim
        cat_embs = [
            self.dropout(self.cat_emb[k](cat[:, :, i]))      # (B, L, D)
            for i, k in enumerate(self.cat_keys)
        ]
        cont_embs = [
            self.dropout(self.cont_lin[k](cont[:, :, i].unsqueeze(-1)))
            for i, k in enumerate(self.cont_keys)
        ]
        # stack: (B, L, K, D)  ->  reshape: (B·L, K, D)
        return torch.stack(cat_embs + cont_embs, dim=2).view(B * L, -1, D)


class FieldTransformer(nn.Module):
    """Intra-row interactions among K fields."""

    def __init__(self, cfg: FieldTransformerConfig):
        super().__init__()
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

    def forward(self, x: Tensor) -> Tensor:           # (B·L, K, D)
        return self.encoder(x)                        # (B·L, K, D)


# -------------------------------------------------------------------------------------- #
#  Sequence transformer (temporal)                                                      #
# -------------------------------------------------------------------------------------- #
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:           # (B, L, M)
        B, L, M = x.shape
        if M != self.d_model:
            raise ValueError("Mismatch in positional encoding dimension.")
        pos = torch.arange(L, device=x.device).float().unsqueeze(1)         # (L, 1)
        div = torch.exp(
            torch.arange(0, M, 2, device=x.device).float() * -(math.log(10000.0) / M)
        )
        pe = torch.zeros(L, M, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)                                         # broadcast B


class SequenceTransformer(nn.Module):
    """Causal Transformer encoder over the temporal dimension."""

    def __init__(self, cfg: SequenceTransformerConfig):
        super().__init__()
        self.pos_enc = SinCosPositionalEncoding(cfg.d_model)
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

    def forward(self, x: Tensor, pad_mask: BoolTensor) -> Tensor:  # (B, L, M)
        B, L, _ = x.shape
        x = self.pos_enc(x)
        causal = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        return self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)


# -------------------------------------------------------------------------------------- #
#  LSTM head                                                                            #
# -------------------------------------------------------------------------------------- #
class LSTMHead(nn.Module):
    """Sequence-level representation ➟ classification or regression."""

    def __init__(self, cfg: LSTMConfig, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, seq_out: Tensor, pad_mask: BoolTensor) -> Tensor:  # (B, L, M)
        lengths = (~pad_mask).sum(dim=1).cpu()
        packed = pack_padded_sequence(seq_out, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])                                          # (B, C)


# -------------------------------------------------------------------------------------- #
#  Full model                                                                           #
# -------------------------------------------------------------------------------------- #
class TransactionModel(nn.Module):
    """End-to-end model supporting autoregressive pre-train and fraud classification."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ─── Embedding & field transformer ─────────────────────────────────────── #
        self.embedder = EmbeddingLayer(
            cfg.cat_vocab_sizes,
            cfg.cont_features,
            emb_dim=cfg.emb_dim,
            dropout=cfg.dropout,
            padding_idx=cfg.padding_idx,
        )
        self.field_tf = FieldTransformer(cfg.field_transformer)

        # ─── Row projection ─────────────────────────────────────────────────────── #
        K, D = len(cfg.cat_vocab_sizes) + len(cfg.cont_features), cfg.emb_dim
        self.row_proj = nn.Linear(K * D, cfg.sequence_transformer.d_model)

        # ─── Sequence transformer ──────────────────────────────────────────────── #
        self.seq_tf = SequenceTransformer(cfg.sequence_transformer)
        

        # ─── heads ─────────────────────────────────────────────── #
        if cfg.lstm_config is not None:
            self.lstm_head = LSTMHead(cfg.lstm_config,
                                      input_size=cfg.sequence_transformer.d_model)
        else:
            self.lstm_head = None     # fraud head skipped
        total_cat = sum(cfg.cat_vocab_sizes.values())
        H = (cfg.lstm_config.hidden_size
             if cfg.lstm_config else cfg.sequence_transformer.d_model)
        self.ar_cat_head  = nn.Linear(H, total_cat)
        self.ar_cont_head = nn.Linear(H, len(cfg.cont_features))
        # pre-compute offsets for categorical slicing
        self.register_buffer(
            "cat_offsets",
            torch.tensor([0] + list(cfg.cat_vocab_sizes.values())).cumsum(0)[:-1],
            persistent=False,
        )

    # ╭──────────────────────────────────────────────────────────────────────────╮ #
    # │                               helpers                                    │ #
    # ╰──────────────────────────────────────────────────────────────────────────╯ #
    def _encode(self, cat: LongTensor, cont: Tensor, pad_mask: BoolTensor) -> Tensor:
        """
        Returns a sequence representation (B, H) for downstream heads.
        """
        B, L, _ = cat.shape
        field = self.embedder(cat, cont)                    # (B·L, K, D)
        field = self.field_tf(field).flatten(start_dim=1)   # (B·L, K*D)
        row_repr = self.row_proj(field).view(B, L, -1)      # (B, L, M)
        seq_out = self.seq_tf(row_repr, pad_mask)           # (B, L, M)

        # LSTM summarisation (same as fraud head uses)
        lengths = (~pad_mask).sum(dim=1).cpu()
        packed = pack_padded_sequence(seq_out, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm_head.lstm(packed)
        return h_n[-1]                                      # (B, H)

    # ╭──────────────────────────────────────────────────────────────────────────╮ #
    # │                                 API                                      │ #
    # ╰──────────────────────────────────────────────────────────────────────────╯ #
    def forward(
        self,
        cat: LongTensor,
        cont: Tensor,
        pad_mask: BoolTensor,
        mode: str = "fraud",
    ):
        """
        Parameters
        ----------
        mode : {'fraud', 'ar'}
            * 'fraud' → classification logits (B, 2)
            * 'ar'    → tuple(logits_cat, pred_cont) for next-step prediction
        """
        hidden = self._encode(cat, cont, pad_mask)          # (B, H)

        if mode == "fraud":
            if self.lstm_head is None:
                raise RuntimeError("Model was built in pre-train-only mode.")
            return self.lstm_head.fc(hidden)
        elif mode == "ar":
            return self.ar_cat_head(hidden), self.ar_cont_head(hidden)
