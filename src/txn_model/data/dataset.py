import logging
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from torch import Tensor
logger = logging.getLogger(__name__)


import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict



# ---------------------------------------------------------------------------
# 1. masking logic stays standalone
# ---------------------------------------------------------------------------
def mask_batch(batched, mask_prob=0.15, row_prob=0.10):
    cat, cont, cont_bin = batched["cat"], batched["cont"], batched["contbin"]
    cat_features = [
        "User",
        "Card",
        "Use Chip",
        "Merchant Name",
        "Merchant City",
        "Merchant State",
        "Zip",
        "MCC",
        "Errors?",
        "Year",
        "Month",
        "Day",
        "Hour",
    ]

    B, T, Fc = cat.shape
    _, _, Fn = cont.shape
    IGN = -100

    # ---------- categorical ----------
    cat_labels = cat.clone()
    mcat = torch.rand_like(cat.float()) < mask_prob
    mcat |= (torch.rand(B, T, 1, device=cat.device) < row_prob)

    # --- 1) build the usual per‑field & per‑row masks -------------
    mcat = (torch.rand_like(cat.float()) < mask_prob)            #   (B,T,Fc)
    mcat |= (torch.rand(B, T, 1, device=cat.device) < row_prob)  # + row masking

    # --- 2) enforce joint masking for time stamp -----------------
    TIME_COLS = ["Year", "Month", "Day", "Hour"]   # keep only the names you really have
    time_idx  = torch.tensor([
        cat_features.index(c) for c in TIME_COLS   # -> e.g. [   9, 10, 11, 12 ]
    ], device=cat.device)

    time_mask = torch.rand(B, T, 1, device=cat.device) < mask_prob   # one bit per (B,T)
    time_mask |= (torch.rand(B, T, 1, device=cat.device) < row_prob) # still allow row mask

    mcat[:, :, time_idx] = time_mask            # broadcast to the four columns




    cat[mcat] = 0
    cat_labels[~mcat] = IGN

    # ---------- numerical ----------
    cont_labels = cont_bin.clone()
    mnum = torch.rand_like(cont.float()) < mask_prob
    mnum |= (torch.rand(B, T, 1, device=cont.device) < row_prob)
    cont[mnum]     = 0.0          # sentinel float
    cont_bin[mnum] = 0
    cont_labels[~mnum] = IGN

    batched.update(
        cat=cat, cont=cont,
        cat_labels=cat_labels,
        cont_labels=cont_labels,
    )
    return batched


# ---------------------------------------------------------------------------
# 2. collate function that DataLoader actually uses
# ---------------------------------------------------------------------------
from typing import List, Dict
import torch

def collate_mlm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    1. Stack per-sample tensors ➜ batched tensors.
    2. Call mask_batch so outputs already include cat_labels & cont_labels.
    """
    cats      = torch.stack([b["cat"]     for b in batch], dim=0)  # [B,T,Fc]
    conts     = torch.stack([b["cont"]    for b in batch], dim=0)  # [B,T,Fn]
    cont_bins = torch.stack([b["contbin"] for b in batch], dim=0)  # [B,T,Fn]

    batched = {"cat": cats, "cont": conts, "contbin": cont_bins}
    return mask_batch(batched)          # returns dict ready for the model



"""
dataset.py
----------
Sliding‑window extraction *and* fast tensor‑dataset loading for subsequent
runs.

Usage pattern
-------------
# ❶ First run: build + save a tensor file (few minutes, once only)
python bert_pretrain.py --build_windows

# ❷ Later runs (default): dataset auto‑detects the .pt file and memory‑maps it
python bert_pretrain.py
"""

import logging, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Helper to build and save all windows once                                #
# --------------------------------------------------------------------------- #
def build_and_save_windows(
    df: pd.DataFrame,
    save_path: Path,
    group_by: str,
    cat_features: List[str],
    cont_features: List[str],
    bin_edges: Dict[str, np.ndarray],
    window: int,
    stride: int = 1,
) -> None:
    """Materialise every sliding window to one tensor file (cat/cont/contbin)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cat_windows, cont_windows, contbin_windows = [], [], []

    # ------------- raw arrays -------------
    cat_arr  = np.ascontiguousarray(df[cat_features].values,  dtype=np.int32)
    cont_arr = np.ascontiguousarray(df[cont_features].values, dtype=np.float32)

    cont_bin_arr = np.stack(
        [np.digitize(df[f].values, bin_edges[f], right=False) + 1
         for f in cont_features], axis=1).astype(np.int32)

    labels   = np.ascontiguousarray(df["is_fraud"].values, dtype=np.int8)

    # ------------- group offsets -------------
    group_sizes = df.groupby(group_by, sort=False).size().to_numpy(dtype=np.int32)
    starts = np.concatenate([[0], group_sizes.cumsum()[:-1]])
    group_offsets = [(int(starts[i]), int(group_sizes[i]))
                     for i in range(len(group_sizes))
                     if group_sizes[i] >= window]

    # ------------- iterate once -------------
    for gidx, (base, length) in enumerate(group_offsets):
        # 1) regular sliding windows
        for off in range(0, length - window + 1, stride):
            st, en = base + off, base + off + window
            cat_windows.append(torch.from_numpy(cat_arr[st:en]).long())
            cont_windows.append(torch.from_numpy(cont_arr[st:en]))
            contbin_windows.append(torch.from_numpy(cont_bin_arr[st:en]).long())

        # 2) include any window whose last tx is fraud
        labels_grp = labels[base: base + length]
        fraud_pos  = np.nonzero(labels_grp == 1)[0]
        for pos in fraud_pos:
            off = int(pos) - (window - 1)
            if 0 <= off <= length - window:
                st, en = base + off, base + off + window
                cat_windows.append(torch.from_numpy(cat_arr[st:en]))
                cont_windows.append(torch.from_numpy(cont_arr[st:en]))
                contbin_windows.append(torch.from_numpy(cont_bin_arr[st:en]))

    logger.info(f"Built {len(cat_windows):,} windows → {save_path.name}")
    torch.save({
        "cat":      torch.stack(cat_windows,     dim=0),   # (N, W, Fc)
        "cont":     torch.stack(cont_windows,    dim=0),   # (N, W, Fn)
        "contbin":  torch.stack(contbin_windows, dim=0),   # (N, W, Fn)
    }, save_path)
    logger.info("Saved windows tensor file.")


# --------------------------------------------------------------------------- #
# 2. Memory‑mapped tensor dataset (fast path)                                 #
# --------------------------------------------------------------------------- #
class TensorWindowDataset(Dataset):
    """Loads the *pre‑built* tensor file via mmap (zero‑copy)."""

    def __init__(self, tensor_path: Path):
        blob = torch.load(tensor_path, mmap=True, weights_only=False)  # zero‑copy
        self.cat      = blob["cat"]      # (N, W, Fc) torch.int32
        self.cont     = blob["cont"]     # (N, W, Fn) torch.float32
        self.contbin  = blob["contbin"]  # (N, W, Fn) torch.int32

    def __len__(self) -> int:
        return self.cat.size(0)

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        return {
            "cat":      self.cat[i],
            "cont":     self.cont[i],
            "contbin":  self.contbin[i],
        }


# --------------------------------------------------------------------------- #
# 3. Fallback sliding‑window dataset (build‑on‑the‑fly)                       #
#    (same as your original, trimmed here for brevity)                        #
# --------------------------------------------------------------------------- #
class TxnDataset(Dataset):
    """Sliding window dataset: 
       - sample windows every `stride`
       - also force include any window whose *last* tx is fraud
       - no window appears twice
    """
    def __init__(
        self,
        df,                     # pandas.DataFrame
        group_by: str,          # e.g. "user_id"
        cat_features: List[str],
        cont_features: List[str],
        bin_edges,
        window: int,
        stride: int = 1,
    ):
        # raw arrays
        self.cat_arr  = np.ascontiguousarray(
            df[cat_features].values, dtype=np.int32
        )
        self.cont_arr = np.ascontiguousarray(
            df[cont_features].values, dtype=np.float32
        )
        self.labels   = np.ascontiguousarray(
            df["is_fraud"].values, dtype=np.int8
        )
        cont_bins = []
        for f in cont_features:
            bins = np.digitize(df[f].values, bin_edges[f], right=False) + 1
            # +1 so 0 stays free for the MASK token
            cont_bins.append(bins.astype(np.int32))

        self.cont_bin_arr = np.ascontiguousarray(np.stack(cont_bins, axis=1))
        self.window   = window
        self.stride   = stride

        # compute group offsets (start index in the flat arrays + group length)
        group_sizes = df.groupby(group_by, sort=False).size().to_numpy(dtype=np.int32)
        starts = np.concatenate([[0], group_sizes.cumsum()[:-1]])
        self.group_offsets = [
            (int(starts[i]), int(group_sizes[i]))
            for i in range(len(group_sizes))
            if group_sizes[i] >= window
        ]

        # build index set
        idx_set = set()
        for gidx, (base, length) in enumerate(self.group_offsets):
            # 1) regular windows every `stride`
            for off in range(0, length - window + 1, stride):
                idx_set.add((gidx, off))

            # 2) force‐include any window ending on a fraud
            labels_grp = self.labels[base : base + length]
            fraud_positions = np.nonzero(labels_grp == 1)[0]
            for pos in fraud_positions:
                off = int(pos) - (window - 1)
                if 0 <= off <= length - window:
                    idx_set.add((gidx, off))

        # turn into an array for __getitem__
        self.indices = np.array(list(idx_set), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        gidx, off = int(self.indices[i, 0]), int(self.indices[i, 1])
        base, _   = self.group_offsets[gidx]
        st, en    = base + off, base + off + self.window

        cat_win  = torch.from_numpy(self.cat_arr[st:en]).long()
        cont_win = torch.from_numpy(self.cont_arr[st:en]).float()
        cont_bin_win = torch.from_numpy(self.cont_bin_arr[st:en]).float()
        # label = last transaction in window
        label    = torch.tensor(int(self.labels[en - 1]), dtype=torch.long)

        return {"cat": cat_win, "cont": cont_win, "contbin": cont_bin_win, "label": label}

# --------------------------------------------------------------------------- #
# 4. Public factory                                                           #
# --------------------------------------------------------------------------- #
def get_dataset(
    df: pd.DataFrame,
    cache_path: Path,
    build: bool,
    **kw,          # all SlidingWindowDataset args except df & cache_path
) -> Dataset:
    """
    Returns a dataset: TensorWindowDataset if cache exists (or we build it now),
    otherwise SlidingWindowDataset.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not build:
        print("loading prebuiltl windows")
        logger.info(f"Loading pre‑built windows from {cache_path.name}")
        return TensorWindowDataset(cache_path)

    if build:
        logger.info("Pre‑building windows tensor file ...")
        build_and_save_windows(df, cache_path, **kw)
        return TensorWindowDataset(cache_path)

    logger.warning("Tensor file not found → falling back to slow dataset.")
    return TxnDataset(df, **kw)


# --------------------------------------------------------------------------- #
# 5. GPU masking + collate                                                   #
# --------------------------------------------------------------------------- #
# dataset.py  --------------------------------------------------
def collate_mlm(batch):
    cat   = torch.stack([b["cat"]     for b in batch])   # CPU
    cont  = torch.stack([b["cont"]    for b in batch])
    cbin  = torch.stack([b["contbin"] for b in batch])
    return {"cat": cat, "cont": cont, "contbin": cbin}   # no labels yet
