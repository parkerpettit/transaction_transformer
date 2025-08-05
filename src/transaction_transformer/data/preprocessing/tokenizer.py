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
        """Assign each value in x to a quantile bin.
        Implements the same semantics as the paper's `np.digitize` + clip.
        Returns indices in [0, num_bins-1].
        """
        x_contiguous = x.contiguous()
        # torch.bucketize behaves like np.digitize when right=False (default)
        idx = torch.bucketize(x_contiguous, self.edges, right=False)
        # idx is in [0, num_edges]; clamp so that values equal to the last edge
        # get mapped to the last valid bin and negatives stay in the first bin.
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        return idx


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

    # Use the paper's approach: create quantiles and remove duplicates
    qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
    edges = np.quantile(series.to_numpy(), qtls)
    edges = np.sort(np.unique(edges))
    
    # Ensure we have at least 2 edges (min and max)
    if len(edges) < 2:
        edges = np.array([series.min(), series.max()])
    
    # Convert to torch tensor
    return NumBinner(edges=torch.from_numpy(edges))


