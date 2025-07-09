from dataclasses import dataclass
@dataclass
class FieldTransformerConfig:
    d_model:        int  # embedding width D for intra-row
    n_heads:        int
    depth:          int
    ffn_mult:       int
    dropout:        float
    layer_norm_eps: float
    norm_first:     bool

@dataclass
class SequenceTransformerConfig:
    d_model:        int  # embedding width m for inter-row
    n_heads:        int
    depth:          int
    ffn_mult:       int
    dropout:        float
    layer_norm_eps: float
    norm_first:     bool

@dataclass
class ModelConfig:
    # embedding
    cat_vocab_sizes: dict[str,int]
    cont_features:    list[str]
    emb_dim:          int
    dropout:          float
    padding_idx:      int

    # intra-row transformer
    field_transformer: FieldTransformerConfig

    # inter-row transformer
    sequence_transformer: SequenceTransformerConfig

    # head
    mlp_hidden:  int
