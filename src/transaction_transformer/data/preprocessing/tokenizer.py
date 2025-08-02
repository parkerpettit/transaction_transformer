"""
Tokenizer for transaction transformer.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Constants
PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
FIRST_REAL_ID = 3


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CatEncoder:
    """Encoder for categorical features."""
    mapping: Dict[Any, int]   # value -> id
    inv: np.ndarray           # id -> token
    pad_id: int = PAD_ID
    mask_id: int = MASK_ID
    unk_id: int = UNK_ID
    first_real_id: int = FIRST_REAL_ID

    @property
    def vocab_size(self) -> int:
        return len(self.inv)

    def is_special(self, ids: np.ndarray) -> np.ndarray:
        return (ids == self.pad_id) | (ids == self.mask_id) | (ids == self.unk_id)


@dataclass
class NumBinner:
    """Binner for numerical features."""
    edges: torch.Tensor  # shape (num_bins+1,), monotonic

    @property
    def num_bins(self) -> int:
        return self.edges.numel() - 1

    def bin(self, x: torch.Tensor) -> torch.Tensor:
        # returns ints in [0 .. num_bins-1]
        # Ensure tensor is contiguous to avoid performance warnings
        x_contiguous = x.contiguous()
        return torch.bucketize(x_contiguous, self.edges, right=True) - 1


@dataclass
class FieldSchema:
    """Schema defining field configurations for preprocessing."""
    cat_features: list[str]          # names in order
    cont_features: list[str]
    cat_encoders: dict[str, CatEncoder]
    cont_binners: dict[str, NumBinner]
    time_cat: list[str] = field(default_factory=list)   # timestamp cat field names
    scaler: StandardScaler = field(default_factory=StandardScaler) # scaler for continuous features
    
    
    # Utilities:
    def cat_idx(self, name: str) -> int: 
        return self.cat_features.index(name)
    
    def cont_idx(self, name: str) -> int: 
        return self.cont_features.index(name)

    


# ============================================================================
# Categorical Encoding Functions
# ============================================================================

def get_encoders(df: pd.DataFrame, cat_features: List[str]) -> Dict[str, CatEncoder]:
    """
    Fit encoders on the provided dataframe.
    Real categories will be assigned ids [3 .. 3+K-1].
    """
    encoders: Dict[str, CatEncoder] = {}
    for c in cat_features:
        cats = pd.Series(df[c], copy=False).astype("category").cat.categories.tolist()
        mapping = {tok: FIRST_REAL_ID + idx for idx, tok in enumerate(cats)}
        inv = np.array(["[PAD]", "[MASK]", "[UNK]", *cats], dtype=object)
        encoders[c] = CatEncoder(mapping=mapping, inv=inv)
    return encoders


def encode_df(df: pd.DataFrame, encoders: Dict[str, CatEncoder], cat_features: List[str]) -> pd.DataFrame:
    """
    Map all categorical values to integer ids.
    Unknowns -> UNK (2). Output dtype is int64-friendly for torch.
    """
    for c in cat_features:
        enc = encoders[c]
        # ensure object so map() returns NaN for unseen
        codes = (
            df[c].astype("object")
                 .map(enc.mapping.get)
                 .fillna(UNK_ID)            # <- UNK, not MASK
                 .astype(np.int64)
        )
        df[c] = codes
    return df


# ============================================================================
# Numerical Processing Functions
# ============================================================================

def get_scaler(df: pd.DataFrame, cont_features: List[str] = ["Amount"]) -> StandardScaler:
    """Fit a StandardScaler on continuous features."""
    return StandardScaler().fit(df[cont_features].to_numpy())


def normalize(df: pd.DataFrame, scaler: StandardScaler, cont_features: List[str] = ["Amount"]) -> pd.DataFrame:
    """Normalize continuous features using the provided scaler."""
    df[cont_features] = scaler.transform(df[cont_features])
    df[cont_features] = df[cont_features].astype(np.float32)
    return df


def build_quantile_binner(series: pd.Series, num_bins: int = 100) -> NumBinner:
    """Build a quantile-based binner for numerical features using numpy for memory efficiency."""

    q = np.linspace(0.0, 1.0, num_bins + 1)
    # Use numpy quantile which is more memory efficient for large datasets
    edges = np.quantile(series.to_numpy(), q)
    edges[0] = float('-inf')
    edges[-1] = float('inf')
    return NumBinner(edges=torch.from_numpy(edges))


