import torch
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import pandas as pd

class TxnDataset(Dataset):
    """
    Dataset for sequence‑to‑next‑row training on transaction data, generating each prefix on-the-fly.


    """
    def __init__(
        self,
        df: pd.DataFrame,
        group_by: str, # cc_num
        cat_features: list[str],
        cont_features: list[str],
        mode: str,
        # max_len: int | None = None,
    ):
      """
      Args:
          df: sorted pandas DataFrame (by group_key then transaction time), with categorical features already
              encoded to integer IDs and continuous features as floats.
          group_key: name of the column to group on (for us, "cc_num")
          cat_features: list of feature names in df
          cont_features: list of continuous feature names in df
          max_len: max content window length. if None, we use full history
      """
      self.user_seqs = []
      self.mode = mode

      for uid, group in df.groupby(group_by): # e.g. group data by cc_num
        cat_tensor = torch.tensor(group[cat_features].to_numpy(), dtype=torch.long)
        cont_tensor = torch.tensor(group[cont_features].to_numpy(), dtype=torch.float)
        self.user_seqs.append((cat_tensor, cont_tensor))

    def __len__(self):
      """
      Returns the number of examples in the dataset (number of user histories)
      """
      return len(self.user_seqs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
      """
      Returns:
              inputs: {
                "cat": LongTensor of shape [seq_len, C],
                "cont": FloatTensor of shape [seq_len, F]
              }
        where C is the number of categorical features and F is the number of continuous features.

      """

      cat_tensor, cont_tensor = self.user_seqs[idx]
      L = cat_tensor.size(0)
      if self.mode == "random":
        low  = min(256, L)
        high = min(512, L)
        win_len = torch.randint(low, high + 1, ()).item()      # () gives a 0-D tensor
        if L > win_len:
            start = torch.randint(0, L - win_len + 1, ()).item()
        else:
            start = 0

        return {"cat": cat_tensor[start:start+win_len, :], "cont": cont_tensor[start:start+win_len, :]}
      elif self.mode == "tail512":
        return {"cat": cat_tensor[-512:], "cont": cont_tensor[-512:]}


def collate_fn(batch, pad_id=0):
    max_len = max(b["cat"].shape[0] for b in batch)

    def _pad(seq, value):
        pad_len = max_len - seq.shape[0]
        return torch.nn.functional.pad(seq, (0, 0, 0, pad_len), value=value)

    cat  = torch.stack([_pad(b["cat"],  pad_id) for b in batch])   # (B, L, C)
    cont = torch.stack([_pad(b["cont"], 0.0)    for b in batch])   # (B, L, F)

    pad_mask = (cat == pad_id).all(dim=-1)        # (B, L)  bool
    return {"cat": cat, "cont": cont, "pad_mask": pad_mask}





class TxnCtxDataset(Dataset):
    """
    One sample = last <= 255 rows ending at index i
    Returns cat/cont tensors **and** the label for row i.
    """
    def __init__(self, df: pd.DataFrame,
                 group_by: str,             # "cc_num"
                 cat_cols: list[str],
                 cont_cols: list[str],
                 keep_neg_every: int = 20): # subsample non-fraud rows
        self.samples = []
        for _, g in df.groupby(group_by, sort=False):
            cat  = torch.tensor(g[cat_cols ].to_numpy(), dtype=torch.long)
            cont = torch.tensor(g[cont_cols].to_numpy(), dtype=torch.float32)
            y    = torch.tensor(g["is_fraud"].to_numpy(), dtype=torch.int64)

            for i in range(len(g)):
                if y[i] == 0 and (i % keep_neg_every):
                    continue                            # down-sample non-fraud
                start = max(0, i - 255)                # keep <= 512 rows
                self.samples.append({
                    "cat"  : cat [start:i+1],
                    "cont" : cont[start:i+1],
                    "label": y[i]
                })

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_ctx(batch, pad_id=0):
    max_len = max(b["cat"].shape[0] for b in batch)

    def _pad(x, value):                    # x : (L, dim)
        pad_len = max_len - x.shape[0]
        return torch.nn.functional.pad(x, (0, 0, 0, pad_len), value=value)

    cat  = torch.stack([_pad(b["cat"],  pad_id) for b in batch])   # B,L,C
    cont = torch.stack([_pad(b["cont"], 0.0)    for b in batch])   # B,L,F
    pad  = (cat == pad_id).all(dim=-1)                            # B,L
    y    = torch.tensor([b["label"] for b in batch], dtype=torch.int64)
    return {"cat": cat, "cont": cont, "pad_mask": pad, "label": y}



