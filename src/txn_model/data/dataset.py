import torch
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
class txnDataset(Dataset):
    """
    Dataset for sequence‑to‑next‑row training on transaction data, generating each prefix on-the-fly.

  
    """
    def __init__(
        self,
        df: pd.DataFrame,
        group_key: str,
        cat_features: list[str],
        cont_features: list[str],
        max_len: int | None = None,
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
      self.max_len = max_len
      self.df = df.reset_index(drop=True)
      self.cat_array = self.df[cat_features].to_numpy(dtype=np.int64)
      self.cont_array = self.df[cont_features].to_numpy(dtype=np.float32)

      self.group_lengths = []
      self.group_starts = []
      for _, grp in self.df.groupby(by=group_key, sort=False):
          start_idx = int(grp.index.min())
          length = int(len(grp))
          if length < 2: # not enough transactions for a sample
              continue
          self.group_starts.append(start_idx)
          self.group_lengths.append(length)
      
      # each group of L transactions creates L - 1 samples
      counts = [L - 1 for L in self.group_lengths]
      self.cum_counts = np.concatenate(([0], np.cumsum(counts)))

    def __len__(self):
      """
      Returns the number of examples in the dataset, i.e. the total
      number of prefixes, which is sum(L-1 for each group).
      """
      return int(self.cum_counts[-1])

    def __getitem__(self, idx: int) -> tuple[
        dict[str, Tensor],  # inputs
        dict[str, Tensor],  # targets
    ]:
      """
      Returns: 
              inputs: {
                "cat": LongTensor of shape [seq_len, C],
                "cont": FloatTensor of shape [seq_len, F]
              }
              targets: {
                "tgt_cat": LongTensor of shape [C],
                "tgt_cont": FloatTensor of shape [F]
              }
        where C is the number of categorical features and F is the number of continuous features.

      """
      # locate which group (card) this idx belongs to
      # cum_counts[i] = total examples before group i
      group_id = int(np.searchsorted(self.cum_counts, idx, side="right",) - 1)

      # index step within that group (0 <= t < group_length - 1)
      t = int(idx - self.cum_counts[group_id])

      base = self.group_starts[group_id]
      seq_end = base + t + 1 # exclusive index of next transaction

      # enforce fixed window
      if self.max_len is not None:
          seq_start = max(base, seq_end - self.max_len)
      else:
          seq_start = base

      cat_context_np = self.cat_array[seq_start:seq_end] # shape (seq_len, C)
      cont_context_np = self.cont_array[seq_start:seq_end] # shape (seq_len, F)

      cat_target_np = self.cat_array[seq_end] # shape (C,)
      cont_target_np = self.cont_array[seq_end] # shape (F,)

      # convert to tensors
      inputs = {
          "cat": torch.from_numpy(cat_context_np).long(),
          "cont": torch.from_numpy(cont_context_np).float()
      }
      targets = {
          "tgt_cat": torch.from_numpy(cat_target_np).long(),
          "tgt_cont": torch.from_numpy(cont_target_np).float()
      }


      return inputs, targets


