from typing import List, Dict, Optional
import torch
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig


class BaseTabCollator:
    """Base collator class for tabular data."""

    def __init__(self, schema: FieldSchema):
        self.schema = schema
        self.cont_binners = {
            name: binner for name, binner in self.schema.cont_binners.items()
        }


class MLMTabCollator(BaseTabCollator):
    """MLM collator (optimized with vectorized operations)."""

    def __init__(
        self,
        config: ModelConfig,
        schema: FieldSchema,
    ):
        super().__init__(schema)
        self.p_field = config.training.p_field
        self.p_row = config.training.p_row
        self.mask_idx = config.data.mask_idx
        self.ignore_idx = config.data.ignore_idx

        # Precompute all constants for efficiency
        self.time_cat_idx = [self.schema.cat_idx(n) for n in self.schema.time_cat]
        self.cat_mask_id = self.mask_idx  # All categorical features use same mask_id
        self.ignore_index = self.ignore_idx

    @torch.no_grad()
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Each item: {"cat": (L,C) long, "cont": (L,F) float, "label": ... optional}
        cats = torch.stack([b["cat"] for b in batch], 0)  # (B,L,C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B,L,F)
        B, L, C = cats.shape
        _, _, F = conts.shape
        device = cats.device

        # 1) Sample field masks (optimized)
        field_mask_cat = torch.rand(B, L, C, device=device) < self.p_field
        field_mask_cont = torch.rand(B, L, F, device=device) < self.p_field

        # 2) Joint timestamp masking (one Bernoulli per (B, L), applied to all time subfields)
        if self.time_cat_idx:
            time_joint = (torch.rand(B, L, device=device) < self.p_field).unsqueeze(
                -1
            )  # (B, L, 1)
            field_mask_cat[..., self.time_cat_idx] |= time_joint.expand(
                -1, -1, len(self.time_cat_idx)
            )

        # 3) Row masking (fused with field masking)
        row_mask = (torch.rand(B, L, device=device) < self.p_row).unsqueeze(
            -1
        )  # (B, L, 1)
        mask_cat = field_mask_cat | row_mask.expand(-1, -1, C)  # (B, L, C)
        mask_cont = field_mask_cont | row_mask.expand(-1, -1, F)  # (B, L, F)

        # Do not mask special categorical tokens: PAD=0, MASK=1, UNK=2
        is_special = (cats == 0) | (cats == 1) | (cats == 2)
        mask_cat = mask_cat & (~is_special)

        # 4) Build labels (vectorized)
        labels_cat = torch.full(
            (B, L, C), fill_value=self.ignore_index, dtype=torch.long, device=device
        )
        labels_cont = torch.full(
            (B, L, F), fill_value=self.ignore_index, dtype=torch.long, device=device
        )

        # Categorical labels: set to true id where masked (vectorized)
        labels_cat[mask_cat] = cats[mask_cat]

        # Continuous labels: batch all binning operations
        # Pre-compute all binned values at once
        all_bins = torch.stack(
            [
                self.schema.cont_binners[name].bin(conts[..., f])
                for f, name in enumerate(self.schema.cont_features)
            ],
            dim=-1,
        )  # (B, L, F)

        # Apply masks vectorized
        labels_cont[mask_cont] = all_bins[mask_cont]

        # 5) Build masked inputs (optimized)
        cats_in = cats.clone()
        cont_in = conts.clone()

        # Vectorized categorical masking (all features use same mask_id)
        cats_in[mask_cat] = self.cat_mask_id

        # Continuous: NaN sentinel for masked inputs (embedder will inject mask vector)
        cont_in[mask_cont] = float("nan")

        return {
            "cat": cats_in,  # (B,L,C) long
            "cont": cont_in,  # (B,L,F) float (NaN where masked)
            "labels_cat": labels_cat,  # (B,L,C) long; ignore_index where not masked
            "labels_cont": labels_cont,  # (B,L,F) long; ignore_index where not masked
        }


class ARTabCollator(BaseTabCollator):
    """Autoregressive collator (optimized with vectorized operations)."""

    def __init__(
        self,
        config: ModelConfig,
        schema: FieldSchema,
    ):
        super().__init__(schema)
        self.ignore_idx = config.data.ignore_idx

    @torch.no_grad()
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Each item: {"cat": (L,C) long, "cont": (L,F) float, "label": ... optional}
        cats = torch.stack([b["cat"] for b in batch], 0)  # (B,L,C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B,L,F)
        B, L, C = cats.shape
        _, _, F = conts.shape
        device = cats.device

        # For AR training: all-steps supervision (predict next row at each position)
        # Inputs are unmasked sequences
        cats_in = cats  # (B, L, C) - full sequence
        cont_in = conts  # (B, L, F) - full sequence

        # Labels are shifted left by 1 (predict row t+1 from rows <= t)
        labels_cat = torch.full(
            (B, L, C), fill_value=self.ignore_idx, dtype=torch.long, device=device
        )
        labels_cont = torch.full(
            (B, L, F), fill_value=self.ignore_idx, dtype=torch.long, device=device
        )

        # Shift categorical labels: labels[:, :-1] = cats[:, 1:]
        labels_cat[:, :-1] = cats[:, 1:]

        # Shift and bin continuous labels
        cont_shifted = conts[:, 1:]  # (B, L-1, F)
        # Bin all continuous values at once
        all_bins = torch.stack(
            [
                self.schema.cont_binners[name].bin(cont_shifted[..., f])
                for f, name in enumerate(self.schema.cont_features)
            ],
            dim=-1,
        )  # (B, L-1, F)
        labels_cont[:, :-1] = all_bins

        # Last position has no target (ignore_idx already set)

        return {
            "cat": cats_in,  # (B,L,C) long - input sequence
            "cont": cont_in,  # (B,L,F) float - input sequence
            "labels_cat": labels_cat,  # (B,L,C) long - shifted targets
            "labels_cont": labels_cont,  # (B,L,F) long - shifted targets (binned)
        }



class FinetuneCollator(BaseTabCollator):
    """Simple collator for finetuning - no masking."""

    @torch.no_grad()
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        cats = torch.stack([b["cat"] for b in batch], 0)  # (B, L, C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B, L, F)
        labels = torch.stack([b["label"] for b in batch], 0)  # (B,)

        return {
            "cat": cats,  # (B, L, C)
            "cont": conts,  # (B, L, F)
            "downstream_label": labels,  # (B,) - fraud labels
        }
