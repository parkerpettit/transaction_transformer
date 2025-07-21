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



def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

    cats = torch.stack([(b["cat"]) for b in batch], dim=0)
    conts = torch.stack([(b["cont"]) for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    return {"cat": cats, "cont": conts, "label": labels}



def slice_batch(batch):
    """
    Returns cat_input, cont_input, cat_target, cont_target, label
    """
    cat, cont, label = batch["cat"], batch["cont"], batch["label"]
    return (
        cat[:, :-1], cont[:, :-1],     # inputs
        cat[:, -1],  cont[:, -1],   # targets
        label                            
    )

def mask_batch(batch, mask_prob=0.15, row_prob=0.10):
    cat, cont, cont_bin = batch["cat"], batch["cont"], batch["contbin"]

    B,T,Fc = cat.shape
    _,_,Fn = cont.shape
    IGN = -100   # ignore index for CrossEntropy

    # ---- categorical ----
    cat_labels = cat.clone()
    mcat = torch.rand_like(cat.float()) < mask_prob
    mcat |= (torch.rand(B,T,1,device=cat.device) < row_prob)
    cat[mcat] = 0
    cat_labels[~mcat] = IGN

    # ---- numerical ----
    cont_labels = cont_bin.clone()
    mnum = torch.rand_like(cont.float()) < mask_prob
    mnum |= (torch.rand(B,T,1,device=cont.device) < row_prob)
    cont[mnum]      = 0.0   # sentinel float (your embedder can learn a vector for it)
    cont_bin[mnum]  = 0
    cont_labels[~mnum] = IGN

    batch.update(
        cat=cat, cont=cont,
        cat_labels=cat_labels,
        cont_labels=cont_labels
    )
    return batch
