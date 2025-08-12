import torch.nn as nn
from typing import List


def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, depth: int, dropout: float = 0.1
) -> nn.Sequential:
    """
    Utility function to build an MLP with the specified depth and dropout.
    """
    layers = []
    if depth > 0:
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth - 1):
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)
