"""
config.py
---------
Centralised dataclass definitions used by *transaction_model.py* and the training
script.  All hyper-parameters live here so that every module pulls from the same
source of truth.

Typical usage
-------------
cfg = ModelConfig(
    cat_vocab_sizes = {"User": 5000, "Card": 200},
    cont_features   = ["Amount"],
    emb_dim         = 48,
    dropout         = 0.10,
    padding_idx     = 0,
    total_epochs    = 10,
    field_transformer = FieldTransformerConfig(
        d_model        = 48,
        n_heads        = 4,
        depth          = 1,
        ffn_mult       = 2,
        dropout        = 0.10,
        layer_norm_eps = 1e-6,
        norm_first     = True,
    ),
    sequence_transformer = SequenceTransformerConfig(
        d_model        = 256,
        n_heads        = 4,
        depth          = 4,
        ffn_mult       = 2,
        dropout        = 0.10,
        layer_norm_eps = 1e-6,
        norm_first     = True,
    ),
    lstm_config = LSTMConfig(
        hidden_size = 256,
        num_layers  = 2,
        num_classes = 2,
        dropout     = 0.10,
    ),
)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────────────────
#  Transformer sub-components
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FieldTransformerConfig:
    d_model:        int   # embedding width D for intra-row attention
    n_heads:        int
    depth:          int
    ffn_mult:       int
    dropout:        float
    layer_norm_eps: float
    norm_first:     bool


@dataclass
class SequenceTransformerConfig:
    d_model:        int   # embedding width M for inter-row attention
    n_heads:        int
    depth:          int
    ffn_mult:       int
    dropout:        float
    layer_norm_eps: float
    norm_first:     bool


# ──────────────────────────────────────────────────────────────────────────────
#  LSTM head
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    hidden_size: int
    num_layers:  int
    num_classes: int
    dropout:     float
    

# ──────────────────────────────────────────────────────────────────────────────
#  Master model config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Embedding layer
    cat_vocab_sizes: Dict[str, int]
    cont_features:   List[str]
    emb_dim:         int
    dropout:         float
    padding_idx:     int

    # Training-loop meta
    total_epochs: int

    # Sub-modules
    field_transformer:     FieldTransformerConfig
    sequence_transformer:  SequenceTransformerConfig
    lstm_config: LSTMConfig | None  # none if pretraining
    # Internal consistency checks ------------------------------------------------
    def __post_init__(self) -> None:
        # emb_dim must match field-level d_model
        if self.emb_dim != self.field_transformer.d_model:
            raise ValueError(
                "`emb_dim` ({}) must equal `field_transformer.d_model` ({})"
                .format(self.emb_dim, self.field_transformer.d_model)
            )

        # The sequence transformer’s output dim should usually match the
        # LSTM hidden size – but only if an LSTM head is present.
        if self.lstm_config is not None:
            if self.sequence_transformer.d_model != self.lstm_config.hidden_size:
                print(
                    "Warning: sequence_transformer.d_model ({}) differs from "
                    "lstm_config.hidden_size ({})."
                    .format(self.sequence_transformer.d_model,
                            self.lstm_config.hidden_size)
                )
