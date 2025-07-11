import logging
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
logger = logging.getLogger(__name__)


class TxnDataset(Dataset):
    """Memory-efficient sliding-window dataset for transactions."""
    def __init__(
        self,
        df: pd.DataFrame,
        group_by: str,
        cat_feats: List[str],
        cont_feats: List[str],
        window_size: int,
        stride: int,
    ):
        # Extract lean NumPy arrays
        self.cat_arr  = np.ascontiguousarray(df[cat_feats].values, dtype=np.int32)
        self.cont_arr = np.ascontiguousarray(df[cont_feats].values, dtype=np.float32)
        self.labels   = np.ascontiguousarray(df["is_fraud"].values, dtype=np.int8)
        self.window_size = window_size
        self.stride      = stride

        # Compute group offsets vectorized
        group_sizes = df.groupby(group_by).size().to_numpy(dtype=np.int32)
        starts = np.concatenate([[0], group_sizes.cumsum()[:-1]])
        offsets = [
            (int(starts[i]), int(group_sizes[i]))
            for i in range(len(group_sizes))
            if group_sizes[i] >= window_size
        ]
        self.group_offsets = offsets
        # Build flat index array
        idxs = []
        for gidx, (base, length) in enumerate(self.group_offsets):
            for off in range(0, length - window_size + 1, stride):
                idxs.append((gidx, off))
        self.indices = np.array(idxs, dtype=np.int32)  # shape [N, 2]

        # Drop original DataFrame reference
        del df

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        gidx, off = int(self.indices[i,0]), int(self.indices[i,1])
        base, _ = self.group_offsets[gidx]
        st, en = base + off, base + off + self.window_size

        cat_win = torch.from_numpy(self.cat_arr[st:en]).long()
        cont_win = torch.from_numpy(self.cont_arr[st:en]).float()
        label = torch.tensor(int(self.labels[en-1]), dtype=torch.long)

        return {"cat": cat_win, "cont": cont_win, "label": label}


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(item["cat"].shape[0] for item in batch)
    def pad(seq: torch.Tensor, v: float) -> torch.Tensor:
        pad_len = max_len - seq.size(0)
        return F.pad(seq, (0, 0, 0, pad_len), value=v)

    cats = torch.stack([pad(b["cat"], pad_id) for b in batch], dim=0)
    conts = torch.stack([pad(b["cont"], 0.0) for b in batch], dim=0)
    pad_mask = (cats == pad_id).all(dim=-1)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    return {"cat": cats, "cont": conts, "pad_mask": pad_mask, "label": labels}
