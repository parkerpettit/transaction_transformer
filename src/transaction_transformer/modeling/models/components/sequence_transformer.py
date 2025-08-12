import math
import torch
import torch.nn as nn
from torch import Tensor

# -------------------------------------------------------------------------------------- #
#  Sequence transformer (temporal)                                                      #
# -------------------------------------------------------------------------------------- #


class SinCosPositionalEncoding(nn.Module):
    """Auto-expanding sinusoidal PE (no fixed max_len)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", torch.empty(0, 0, d_model), persistent=False)

    def _build(self, max_len: int, device, dtype):
        pos = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )  # (D//2,)
        pe = torch.zeros(max_len, self.d_model, device=device, dtype=dtype)  # (L,D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}")
        if (
            self.pe.numel() == 0
            or self.pe.size(1) < L
            or self.pe.device != x.device
            or self.pe.dtype != x.dtype
        ):
            self._build(L, x.device, x.dtype)
        return x + self.pe[:, :L]


class SequenceTransformer(nn.Module):
    """Transformer encoder over time with optional causal (AR) masking."""

    def __init__(self, config, causal: bool):
        super().__init__()
        self.pos_enc = SinCosPositionalEncoding(config.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_mult * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,  # expect (B, L, M)
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)

        # Auto-expanding causal mask cache
        self.is_causal = causal
        self.register_buffer(
            "causal_mask", torch.empty(0, 0, dtype=torch.bool), persistent=False
        )

    @torch.no_grad()
    def _get_causal_mask(self, L: int, device, dtype=torch.bool) -> Tensor:
        m = self.causal_mask
        if m.numel() == 0 or m.size(0) < L or m.device != device or m.dtype != dtype:
            # True entries are *masked* positions (upper triangle)
            m = torch.triu(torch.ones(L, L, device=device, dtype=dtype), diagonal=1)
            self.causal_mask = m
        return self.causal_mask[:L, :L]

    def forward(
        self,
        x: Tensor,  # (B, L, M)
        *,
        key_padding_mask: Tensor | None = None,  # (B, L) bool: True = pad to ignore
    ) -> Tensor:
        """
        Returns: (B, L, M)

        Args:
            x: Input tensor (B, L, M)
            causal: Whether to use causal masking (for AR training)
            key_padding_mask: Padding mask (B, L) with True entries masked
        """
        B, L, M = x.shape
        x = self.pos_enc(x)

        # Use causal mask if requested and available
        attn_mask = None
        if self.is_causal:
            attn_mask = self._get_causal_mask(L, x.device)

        # nn.TransformerEncoder expects:
        #   mask: (L, L) with True entries masked (bool) or -inf (float)
        #   src_key_padding_mask: (B, L) with True entries masked (ignored)
        return self.encoder(
            x, mask=attn_mask, src_key_padding_mask=key_padding_mask
        )  # (B, L, M)
