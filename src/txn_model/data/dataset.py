import logging
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TxnDataset(Dataset):
    """Sliding-window dataset over each user's transaction history."""

    def __init__(
        self,
        df: pd.DataFrame,
        group_by: str,
        cat_features: List[str],
        cont_features: List[str],
        window_size: int,
        stride: int,
    ) -> None:
        start = torch.cuda.Event(enable_timing=False)
        logger.info(
            "Initializing TxnDataset with window_size=%s stride=%s", window_size, stride
        )
        self.samples: List[Dict[str, torch.Tensor]] = []
        for key, group in df.groupby(group_by, sort=False):
            logger.debug("Processing group %s with %d rows", key, len(group))
            cat_tensor = torch.tensor(group[cat_features].values, dtype=torch.long)
            cont_tensor = torch.tensor(group[cont_features].values, dtype=torch.float)
            label_tensor = torch.tensor(group["is_fraud"].values, dtype=torch.long)
            L = cat_tensor.size(0)
            for st in range(0, L - window_size + 1, stride):
                en = st + window_size
                self.samples.append(
                    {
                        "cat": cat_tensor[st:en],
                        "cont": cont_tensor[st:en],
                        "label": label_tensor[en - 1],
                    }
                )
        logger.info("Built %d samples for TxnDataset", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        logger.debug("Fetching sample %d", idx)
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad variable length sequences in batch."""
    max_len = max(item["cat"].shape[0] for item in batch)
    logger.debug("collate_fn called with batch size %d", len(batch))

    def _pad(seq: torch.Tensor, value: float) -> torch.Tensor:
        pad_len = max_len - seq.shape[0]
        logger.debug("Padding sequence from %d to %d", seq.shape[0], max_len)
        return F.pad(seq, (0, 0, 0, pad_len), value=value)

    cat = torch.stack([_pad(b["cat"], pad_id) for b in batch], dim=0)
    cont = torch.stack([_pad(b["cont"], 0.0) for b in batch], dim=0)
    pad_mask = (cat == pad_id).all(dim=-1)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    logger.debug(
        "Collated batch with max_len=%d -> cat %s cont %s", max_len, tuple(cat.shape), tuple(cont.shape)
    )
    return {"cat": cat, "cont": cont, "pad_mask": pad_mask, "label": labels}

