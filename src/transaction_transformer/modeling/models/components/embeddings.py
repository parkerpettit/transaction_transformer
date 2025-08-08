import torch
import torch.nn as nn
from typing import Dict, List
from transaction_transformer.data.preprocessing.schema import FieldSchema
from transaction_transformer.config.config import ModelConfig

# -------------------------------------------------------------------------------------- #
#  Frequency encoding                                                                    #
# -------------------------------------------------------------------------------------- #
class FrequencyEncoder(nn.Module):
    """
    gamma(v) = [sin(2^0 * pi * v), cos(2^0 * pi * v), ..., sin(2^{L-1} * pi * v), cos(2^{L-1} * pi * v)]
    Outputs last dim = 2*L, interleaved as [sin_0, cos_0, sin_1, cos_1, ...]. Assumes v is already normalized appropriately.
    """
    def __init__(self, L: int = 8):
        super().__init__()
        # Only store what is needed for forward: L and freqs buffer
        self.L = L
        self.register_buffer(
            "freqs",
            (2.0 ** torch.arange(L, dtype=torch.float32)) * torch.pi
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, L, F), match device/dtype of freqs
        v = v.to(self.freqs)  # type: ignore
        x = v.unsqueeze(-1) * self.freqs  # type: ignore (B, L, F, L)
        sin = torch.sin(x)  # (B, L, F, L)
        cos = torch.cos(x)  # (B, L, F, L)
        # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
        return torch.stack((sin, cos), dim=-1).reshape(*x.shape[:-1], 2 * self.L)  # (B, L, F, 2L)

# -------------------------------------------------------------------------------------- #
#  Embedding + field transformer                                                         #
# -------------------------------------------------------------------------------------- #
class EmbeddingLayer(nn.Module):
    """
    Embeds categorical fields and frequency-encodes continuous fields, with masking sentinels.
    
    cat: (B, L, C)  longs with specials (PAD=0, MASK=1, UNK=2, real>=3)
    cont: (B, L, F) floats; masked positions are NaN
    Returns: (B*L, K, D) with K=C+F. Conceptually, we end up with a representation
    of each field in every row.
    """
    def __init__(
        self,
        schema: FieldSchema,
        config: ModelConfig,
        L: int = 8,
    ) -> None:
        super().__init__()
        self.cat_encoders = schema.cat_encoders
        self.cont_binners = schema.cont_binners
        self.cat_features = schema.cat_features
        self.cont_features = schema.cont_features
        self.emb_dim = config.embedding.emb_dim
        self.padding_idx = config.data.padding_idx
        self.dropout_float = config.emb_dropout

        # One learned mask vector for continuous masked positions
        self.mask_token = nn.Parameter(torch.randn(self.emb_dim) * 0.02)

        # Categorical embeddings (order = dict insertion order)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(self.cat_encoders[f].vocab_size, self.emb_dim, padding_idx=self.padding_idx) for f in self.cat_features]
        )

        # Continuous projections: per-feature projection after frequency encoding
        self.freq_enc = FrequencyEncoder(L)
        self.cont_proj = nn.ModuleList(
            [nn.Linear(2 * self.freq_enc.L, self.emb_dim, bias=False) for _ in self.cont_features]
        )

        # Dropout
        self.dropout = nn.Dropout(self.dropout_float) if self.dropout_float > 0 else nn.Identity()
        self.num_fields = len(self.cat_features) + len(self.cont_features)

    def forward(self, cat: torch.LongTensor, cont: torch.Tensor) -> torch.Tensor:
        B, L, _ = cat.shape

        # --- Categorical fields ---
        cat_embs = [emb(cat[..., i]) for i, emb in enumerate(self.cat_emb)]  # each (B, L, D)

        # --- Continuous fields ---
        cont_embs = []
        for i, proj in enumerate(self.cont_proj):
            vals = cont[..., i]                      # (B, L)
            m = torch.isnan(vals)                   # (B, L)
            clean = torch.where(m, torch.zeros_like(vals), vals)
            gamma = self.freq_enc(clean)            # (B, L, 2L)
            y = proj(gamma)                         # (B, L, D)
            if m.any():
                y[m] = self.mask_token             # replace masked slots
            cont_embs.append(y)

        # --- Stack all fields along K ---
        all_embs = cat_embs + cont_embs            # list of (B, L, D) of length K
        embs_stacked = torch.stack(all_embs, dim=2)  # (B, L, K, D)
        embs_stacked = self.dropout(embs_stacked)
        return embs_stacked.reshape(B*L, self.num_fields, self.emb_dim) # (B*L, K, D),
