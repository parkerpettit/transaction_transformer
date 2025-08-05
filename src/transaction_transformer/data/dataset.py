import logging
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from torch import Tensor
logger = logging.getLogger(__name__)
from pathlib import Path
from tqdm.auto import tqdm
from transaction_transformer.data.preprocessing import FieldSchema
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict

class TxnDataset(Dataset):
    """Sliding window dataset"""

    def __init__(
        self,
        df,                     # pandas.DataFrame
        group_by: str,          # e.g. "user_id"
        schema: FieldSchema,
        window: int,
        stride: int = 1,
        include_all_fraud: bool = False,
    ):
        # 1) convert once to torch and pin into shared memory
        cat_np   = df[schema.cat_features].to_numpy(dtype=np.int64)
        cont_np  = df[schema.cont_features].to_numpy(dtype=np.float32)
        label_np = df["is_fraud"].to_numpy(dtype=np.int64)

        self.cat_arr   = torch.from_numpy(cat_np).share_memory_()
        self.cont_arr  = torch.from_numpy(cont_np).share_memory_()
        self.labels    = torch.from_numpy(label_np).share_memory_()
        self.window    = window
        self.stride    = stride

        # 2) compute group offsets
        group_sizes = df.groupby(group_by, sort=False).size().to_numpy(dtype=np.int32)
        starts = np.concatenate([[0], group_sizes.cumsum()[:-1]])
        self.group_offsets = [
            (int(starts[i]), int(group_sizes[i]))
            for i in range(len(group_sizes))
            if group_sizes[i] >= window
        ]

        # 3) build your sliding‐window index list
        idx_set = set()
        for gidx, (base, length) in enumerate(self.group_offsets):
            for off in range(0, length - window + 1, stride):
                idx_set.add((gidx, off))
            labels_grp = self.labels[base : base + length].numpy()
            if include_all_fraud: # include all fraud samples, may lead to overlapping windows
                for pos in np.nonzero(labels_grp == 1)[0]:
                    off = int(pos) - (window - 1)
                    if 0 <= off <= length - window:
                        idx_set.add((gidx, off))
        

        self.indices = torch.as_tensor(list(idx_set), dtype=torch.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # no Python slicing of numpy, just index into shared‐memory torch.Tensors
        gidx, off = int(self.indices[i,0]), int(self.indices[i,1])
        base, _   = self.group_offsets[gidx]
        st, en    = base + off, base + off + self.window

        cat_win  = self.cat_arr[st:en]        # (window, C), Int64Tensor
        cont_win = self.cont_arr[st:en]       # (window, F), FloatTensor
        label    = self.labels[en-1]          # scalar Int64Tensor

        return {"cat": cat_win, "cont": cont_win, "label": label}










# # embed.py
# import numpy as np
# import torch
# from pathlib import Path
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from torch import Tensor

# @torch.inference_mode()
# def extract_embeddings(
#     pretrained_model: torch.nn.Module,
#     ds: TxnDataset,
#     cache_dir: Path,
#     batch_size: int,
#     device: str = "cpu",
# ):
#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     # 1) metadata
#     N = len(ds)
#     M = pretrained_model.config.seq_config.d_model # type: ignore
#     # save meta
#     torch.save({"N": N, "M": M}, cache_dir / "meta.pt")

#     # 2) open memmaps
#     pred_mm = np.memmap(cache_dir / "pred.dat",   dtype="float32", model_type="w+", shape=(N, M)) # type: ignore
#     act_mm  = np.memmap(cache_dir / "actual.dat", dtype="float32", model_type="w+", shape=(N, M))# type: ignore
#     delta_mm= np.memmap(cache_dir / "delta.dat",  dtype="float32", model_type="w+", shape=(N, M))# type: ignore
#     lbl_mm  = np.memmap(cache_dir / "label.dat",  dtype="float32", model_type="w+", shape=(N,))

#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     pretrained_model.to(device).eval()

#     idx = 0
#     for batch in tqdm(loader, desc="Extracting embeddings", ncols=80):
#         B = batch["cat"].size(0)
#         cat, cont, label = (t.to(device) for t in (batch["cat"], batch["cont"], batch["label"]))

#         # predicted embedding from history only
#         h_pred = pretrained_model(cat[:, :-1], cont[:, :-1], model_type="extract").cpu().numpy()
#         # actual embedding of full window
#         h_act  = pretrained_model(cat, cont, model_type="extract").cpu().numpy()

#         pred_mm [idx:idx+B] = h_pred
#         act_mm  [idx:idx+B] = h_act
#         delta_mm[idx:idx+B] = (h_act - h_pred)
#         lbl_mm  [idx:idx+B] =  label.cpu().numpy()

#         idx += B

#     assert idx == N

#     # flush to disk
#     pred_mm.flush()
#     act_mm .flush()
#     delta_mm.flush()
#     lbl_mm .flush()

# # embedded_dataset.py
# import numpy as np
# import torch
# from pathlib import Path
# from torch.utils.data import Dataset
# import numpy as np
# import torch
# from pathlib import Path
# from torch.utils.data import Dataset

# class EmbeddingMemmapDataset(Dataset):
#     def __init__(self, cache_dir: str):
#         cache_dir = Path(cache_dir)
#         meta = torch.load(cache_dir / "meta.pt", map_location="cpu")
#         self.N = int(meta["N"])
#         self.M = int(meta["M"])

#         # Just store filenames (picklable)
#         self._cache_dir = cache_dir
#         self._pred_file   = str(cache_dir / "pred.dat")
#         self._actual_file = str(cache_dir / "actual.dat")
#         self._delta_file  = str(cache_dir / "delta.dat")
#         self._label_file  = str(cache_dir / "label.dat")

#         # Will be replaced by real memmaps in each worker
#         self.predals = None
#         self.actual = None
#         self.delta  = None
#         self.labels = None

#     def __len__(self):
#         return self.N

#     def __getitem__(self, i):
#         # On first access in each worker, open the memmaps
#         if self.predals is None:
#             self.predals = np.memmap(self._pred_file,   dtype="float32",
#                                     model_type="r+", shape=(self.N, self.M))
#             self.actual  = np.memmap(self._actual_file, dtype="float32",
#                                     model_type="r+", shape=(self.N, self.M))
#             self.delta   = np.memmap(self._delta_file,  dtype="float32",
#                                     model_type="r+", shape=(self.N, self.M))
#             self.labels  = np.memmap(self._label_file,  dtype="float32",
#                                     model_type="r+", shape=(self.N,))

#         # Now indexing is just a cheap page‑in by the OS
#         pred = torch.from_numpy(self.predals[i])
#         act  = torch.from_numpy(self.actual[i])
#         d    = torch.from_numpy(self.delta[i])
#         lbl  = torch.tensor(self.labels[i])
#         return {
#             "pred_embedding":   pred,
#             "actual_embedding": act,
#             "delta":            d,
#             "label":            lbl,
#         }

    




    





    


