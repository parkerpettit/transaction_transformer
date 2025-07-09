import torch
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pandas as pd

class TxnDataset(Dataset):
    """
    Sliding‐window dataset over each user's transaction history.

    Each sample is:
      - "cat": LongTensor [window_size, C]
      - "cont": FloatTensor [window_size, F]
      - "label": LongTensor  (0 or 1)  ← the is_fraud flag of the last row in the window
    """
    def __init__(
        self,
        df: pd.DataFrame,
        group_by: str,                   # e.g. "cc_num"
        cat_features: list[str],
        cont_features: list[str],
        window_size: int,
        stride: int,
    ):
        self.samples = []
        # assume df is already sorted by (group_by, time)
        for _, group in df.groupby(group_by):
            cat_tensor   = torch.tensor(group[cat_features].values, dtype=torch.long)
            cont_tensor  = torch.tensor(group[cont_features].values, dtype=torch.float)
            label_tensor = torch.tensor(group["is_fraud"].values, dtype=torch.long)
            L = cat_tensor.size(0)

            # collect every full window
            for start in range(0, L - window_size + 1, stride):
                end = start + window_size
                self.samples.append({
                    "cat":   cat_tensor[start:end],       # [window_size, C]
                    "cont":  cont_tensor[start:end],      # [window_size, F]
                    "label": label_tensor[end - 1],       # scalar
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch: list[dict], pad_id: int = 0):
    """
    Pads a list of {"cat", "cont", "label"} dicts into batched tensors.
    """
    # find the longest sequence in this batch
    max_len = max(item["cat"].shape[0] for item in batch)

    def _pad(seq: torch.Tensor, pad_value: float):
        # pad: (left, right, top, bottom) for 2D: here (0,0) on last dim, (0, pad_len) on first
        pad_len = max_len - seq.shape[0]
        return F.pad(seq, (0, 0, 0, pad_len), value=pad_value)

    cat   = torch.stack([_pad(item["cat"], pad_id) for item in batch], dim=0)  # (B, L, C)
    cont  = torch.stack([_pad(item["cont"], 0.0)  for item in batch], dim=0)  # (B, L, F)
    pad_mask = (cat == pad_id).all(dim=-1)                                      # (B, L)
    labels   = torch.stack([item["label"] for item in batch], dim=0)           # (B,)

    return {
        "cat":      cat,
        "cont":     cont,
        "pad_mask": pad_mask,
        "label":    labels,
    }





# class TxnCtxDataset(Dataset):
#     """
#     One sample = last <= 255 rows ending at index i
#     Returns cat/cont tensors **and** the label for row i.
#     """
#     def __init__(self, df: pd.DataFrame,
#                  group_by: str,             # "cc_num"
#                  cat_cols: list[str],
#                  cont_cols: list[str],
#                  keep_neg_every: int = 20): # subsample non-fraud rows
#         self.samples = []
#         for _, g in df.groupby(group_by, sort=False):
#             cat  = torch.tensor(g[cat_cols ].to_numpy(), dtype=torch.long)
#             cont = torch.tensor(g[cont_cols].to_numpy(), dtype=torch.float32)
#             y    = torch.tensor(g["is_fraud"].to_numpy(), dtype=torch.int64)

#             for i in range(len(g)):
#                 if y[i] == 0 and (i % keep_neg_every):
#                     continue                            # down-sample non-fraud
#                 start = max(0, i - 255)                # keep <= 512 rows
#                 self.samples.append({
#                     "cat"  : cat [start:i+1],
#                     "cont" : cont[start:i+1],
#                     "label": y[i]
#                 })

#     def __len__(self):  return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]


# def collate_fn_ctx(batch, pad_id=0):
#     """Collate samples for context datasets with labels.

#     Pads variable length sequences and stacks them into batch tensors, also
#     returning the fraud label for the last transaction in each sample.
#     """
#     max_len = max(b["cat"].shape[0] for b in batch)

#     def _pad(x, value):                    # x : (L, dim)
#         pad_len = max_len - x.shape[0]
#         return torch.nn.functional.pad(x, (0, 0, 0, pad_len), value=value)

#     cat  = torch.stack([_pad(b["cat"],  pad_id) for b in batch])   # B,L,C
#     cont = torch.stack([_pad(b["cont"], 0.0)    for b in batch])   # B,L,F
#     pad  = (cat == pad_id).all(dim=-1)                            # B,L
#     y    = torch.tensor([b["label"] for b in batch], dtype=torch.int64)
#     return {"cat": cat, "cont": cont, "pad_mask": pad, "label": y}



