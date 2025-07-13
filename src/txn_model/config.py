from dataclasses import dataclass
from typing    import Dict, List, Optional

@dataclass
class TransformerConfig:
    d_model:         int = 48
    n_heads:         int = 4
    depth:           int = 1
    ffn_mult:        int = 2
    dropout:         float = 0.1
    layer_norm_eps:  float = 1e-6
    norm_first:      bool  = True

@dataclass
class LSTMConfig:
    hidden_size:     int   = 256
    num_layers:      int   = 2
    num_classes:     int   = 2
    dropout:         float = 0.1

@dataclass
class ModelConfig:
    cat_vocab_sizes: Dict[str, int]      
    cont_features:   List[str] 
    ft_config:       TransformerConfig     
    seq_config:      TransformerConfig
    lstm_config:     Optional[LSTMConfig] = None
    emb_dropout:     float                = 0.1
    clf_dropout:     float                = 0.1
    padding_idx:     int                  = 0
    total_epochs:    int                  = 10
    window:          int                  = 10
    stride:          int                  = 5
  

