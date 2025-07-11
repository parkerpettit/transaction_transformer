import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor
from torch.nn.utils.rnn import pack_padded_sequence
from config import ModelConfig
import math

class EmbeddingLayer(nn.Module):
    """
    Embedding layer for tabular transactions.

    - Embeds each of C categorical fields via individual nn.Embedding tables
    - Projects F continuous fields via individual nn.Linear layers
    - Returns a (B*L, K, D) tensor where K is the total number of features (C + F).
    """
    def __init__(self,
                 cat_vocab_sizes: dict[str, int],
                 cont_features: list[str],
                 emb_dim: int,
                 dropout: float,
                 padding_idx: int):
        super().__init__()
        self.cat_embeds = nn.ModuleDict({
            f: nn.Embedding(cat_vocab_sizes[f], emb_dim, padding_idx)
            for f in cat_vocab_sizes
        })
        self.cont_proj = nn.ModuleDict({
            f: nn.Linear(1, emb_dim)
            for f in cont_features
        })
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, cat: LongTensor, cont: Tensor) -> Tensor:
        B, L, _ = cat.shape
        D = next(iter(self.cat_embeds.values())).embedding_dim
        print(f"[Model] EmbeddingLayer.forward B={B} L={L}")

        cat_embs = [self.dropout(self.cat_embeds[f](cat[:, :, i]))
                    for i, f in enumerate(self.cat_embeds)]
        cont_embs = [self.dropout(self.cont_proj[f](cont[:, :, i].unsqueeze(-1)))
                     for i, f in enumerate(self.cont_proj)]

        x = torch.stack(cat_embs + cont_embs, dim=2)  # (B, L, K, D)
        return x.reshape(B * L, -1, D)                # (B*L, K, D)

class FieldTransformer(nn.Module):
    """
    Intra-row field interaction via TransformerEncoder
    """
    def __init__(self, d_model: int, n_heads: int, depth: int = 2,
                 ffn_mult: int = 4, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout, activation="relu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True, norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*L, K, D)
        print(f"[Model] FieldTransformer.forward input {tuple(x.shape)}")
        return self.encoder(x)  # (B*L, K, D)

class PositionalEncoding(nn.Module):
    """Sin-cos positional encoding"""
    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        device = x.device
        pos = torch.arange(L, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device).float() * -(math.log(10000.0) / D)
        )
        pe = torch.zeros(L, D, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return x + pe.unsqueeze(0)

class SequenceTransformer(nn.Module):
    """Inter-row (temporal) Transformer encoder with causal masking"""
    def __init__(self, d_model: int, n_heads: int, depth: int = 2,
                 ffn_mult: int = 4, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False):
        super().__init__()
        self.pos_encoder = PositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout, activation="relu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True, norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

    def forward(self, x: Tensor, padding_mask: BoolTensor) -> Tensor:
        # x: (B, L, m)
        print(f"[Model] SequenceTransformer.forward input {tuple(x.shape)}")
        B, L, _ = x.shape
        x = self.pos_encoder(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device
        )
        return self.transformer(
            x,
            mask=causal_mask.bool(),
            src_key_padding_mask=padding_mask.bool()
        )  # (B, L, m)

class LSTMClassifier(nn.Module):
    """LSTM on top of sequence outputs for classification"""
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, padding_mask: BoolTensor) -> Tensor:
        # x: (B, L, m)
        print(f"[Model] LSTMClassifier.forward input {tuple(x.shape)}")
        lengths = (~padding_mask).sum(dim=1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # (B, hidden_size)
        return self.fc(h_last)

class TransactionModel(nn.Module):
    """End-to-end tabular sequence model"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        # dimensions
        C = len(config.cat_vocab_sizes)
        F = len(config.cont_features)
        K = C + F
        D = config.emb_dim
        M = config.sequence_transformer.d_model

        # components
        self.embedder = EmbeddingLayer(
            config.cat_vocab_sizes,
            config.cont_features,
            emb_dim=D,
            dropout=config.dropout,
            padding_idx=config.padding_idx
        )
        ft = config.field_transformer
        self.field_tfmr = FieldTransformer(
            d_model=ft.d_model,
            n_heads=ft.n_heads,
            depth=ft.depth,
            ffn_mult=ft.ffn_mult,
            dropout=ft.dropout,
            layer_norm_eps=ft.layer_norm_eps,
            norm_first=ft.norm_first
        )
        st = config.sequence_transformer
        self.seq_tfmr = SequenceTransformer(
            d_model=st.d_model,
            n_heads=st.n_heads,
            depth=st.depth,
            ffn_mult=st.ffn_mult,
            dropout=st.dropout,
            layer_norm_eps=st.layer_norm_eps,
            norm_first=st.norm_first
        )
        # simple row projection: (K*D) -> M
        self.row_projector = nn.Linear(K * D, M)
        # LSTM classifier (fraud head)
        lstm_cfg = config.lstm_config
        self.fraud_head = LSTMClassifier(
            input_size=M,
            hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers,
            num_classes=lstm_cfg.num_classes,
            dropout=lstm_cfg.dropout
        )

        hidden_dim = lstm_cfg.hidden_size
        self.total_cat_size = sum(config.cat_vocab_sizes.values())
        self.cat_offsets = torch.cumsum(
            torch.tensor([0] + list(config.cat_vocab_sizes.values())), dim=0
        )[:-1]
        self.ar_cat_head = nn.Linear(hidden_dim, self.total_cat_size)
        self.ar_cont_head = nn.Linear(hidden_dim, len(config.cont_features))

    def _encode(self, cat: LongTensor, cont: Tensor, padding_mask: BoolTensor) -> Tensor:
        print(f"[Model] _encode with cat {tuple(cat.shape)} cont {tuple(cont.shape)}")
        B, L, _ = cat.shape
        emb = self.embedder(cat, cont)          # (B*L, K, D)
        intra = self.field_tfmr(emb)            # (B*L, K, D)
        flat = intra.flatten(start_dim=1)       # (B*L, K*D)
        row_repr = self.row_projector(flat).view(B, L, -1)  # (B, L, M)
        seq_out = self.seq_tfmr(row_repr, padding_mask)     # (B, L, M)
        lengths = (~padding_mask).sum(dim=1)
        packed = pack_padded_sequence(
            seq_out, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.fraud_head.lstm(packed)
        return h_n[-1]  # (B, hidden_dim)

    def forward(self,
                cat: LongTensor,
                cont: Tensor,
                padding_mask: BoolTensor,
                mode: str = 'fraud'):
        hidden = self._encode(cat, cont, padding_mask)
        print(f"[Model] forward mode={mode} hidden {tuple(hidden.shape)}")
        if mode == 'ar':
            logits_cat = self.ar_cat_head(hidden)
            pred_cont = self.ar_cont_head(hidden)
            return logits_cat, pred_cont
        else:
            return self.fraud_head.fc(hidden)
