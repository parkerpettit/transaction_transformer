from typing import List, Dict, Optional
import torch
from transaction_transformer.data.preprocessing import FieldSchema
from transaction_transformer.config.config import ModelConfig

class BaseTabCollator:
    """Base collator class for tabular data."""
    
    def __init__(self, schema: FieldSchema):
        self.schema = schema
        self.cont_binners = {name: binner for name, binner in self.schema.cont_binners.items()}

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
        cats  = torch.stack([b["cat"]  for b in batch], 0)  # (B,L,C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B,L,F)
        labels = torch.stack([b["label"] for b in batch], 0) # (B,L)
        B, L, C = cats.shape
        _, _, F = conts.shape
        device = cats.device

        # 1) Sample field masks (optimized)
        field_mask_cat = (torch.rand(B, L, C, device=device) < self.p_field)
        field_mask_cont = (torch.rand(B, L, F, device=device) < self.p_field)

        # 2) Joint timestamp masking (vectorized)
        if self.time_cat_idx:
            time_cat_mask = field_mask_cat[..., self.time_cat_idx]  # (B, L, num_time_cat)
            any_time_masked = time_cat_mask.any(dim=-1, keepdim=True)  # (B, L, 1)
            field_mask_cat[..., self.time_cat_idx] |= any_time_masked.expand(-1, -1, len(self.time_cat_idx))

        # 3) Row masking (fused with field masking)
        row_mask = (torch.rand(B, L, device=device) < self.p_row).unsqueeze(-1)  # (B, L, 1)
        mask_cat = field_mask_cat | row_mask.expand(-1, -1, C)  # (B, L, C)
        mask_cont = field_mask_cont | row_mask.expand(-1, -1, F)  # (B, L, F)

        # 4) Build labels (vectorized)
        labels_cat = torch.full((B, L, C), fill_value=self.ignore_index, dtype=torch.long, device=device)
        labels_cont = torch.full((B, L, F), fill_value=self.ignore_index, dtype=torch.long, device=device)

        # Categorical labels: set to true id where masked (vectorized)
        labels_cat[mask_cat] = cats[mask_cat]

        # Continuous labels: batch all binning operations
        # Pre-compute all binned values at once
        all_bins = torch.stack([
            self.schema.cont_binners[name].bin(conts[..., f]) 
            for f, name in enumerate(self.schema.cont_features)
        ], dim=-1)  # (B, L, F)
        
        # Apply masks vectorized
        labels_cont[mask_cont] = all_bins[mask_cont]

        # 5) Build masked inputs (optimized)
        cats_in = cats.clone()
        cont_in = conts.clone()

        # Vectorized categorical masking (all features use same mask_id)
        cats_in[mask_cat] = self.cat_mask_id

        # Continuous: NaN sentinel for masked inputs (embedder will inject mask vector)
        cont_in[mask_cont] = float('nan')

        return {
            "cat": cats_in,                 # (B,L,C) long
            "cont": cont_in,                # (B,L,F) float (NaN where masked)
            "labels_cat": labels_cat,       # (B,L,C) long; ignore_index where not masked 
            "labels_cont": labels_cont,     # (B,L,F) long; ignore_index where not masked
            "downstream_label": labels,     # passthrough if you need it
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
        cats  = torch.stack([b["cat"]  for b in batch], 0)  # (B,L,C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B,L,F)
        labels = torch.stack([b["label"] for b in batch], 0)  # (B,L)

        # For AR training: use first L-1 transactions as input, predict transaction L
        cats_in = cats[:, :-1, :]  # (B, L-1, C) - input sequence
        cont_in = conts[:, :-1, :]  # (B, L-1, F) - input sequence
        
        # Labels are the last transaction (vectorized binning)
        labels_cat = cats[:, -1, :]  # (B, C) - last transaction
        labels_cont = conts[:, -1, :]  # (B, F) - last transaction
        
        # Vectorized continuous binning
        labels_cont = torch.stack([
            self.schema.cont_binners[name].bin(labels_cont[:, f]) 
            for f, name in enumerate(self.schema.cont_features)
        ], dim=1)  # (B, F)
        
        return {
            "cat": cats_in,                 # (B,L-1,C) long - input sequence
            "cont": cont_in,                # (B,L-1,F) float - input sequence
            "labels_cat": labels_cat,       # (B,C) long - target transaction
            "labels_cont": labels_cont,     # (B,F) long - target transaction (binned values)
            "downstream_label": labels,     # (B,L) - passthrough if you need it
        }


# Keep the old function for backward compatibility
def collate_fn_autoregressive(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Legacy autoregressive collator function."""
    cats = torch.stack([(b["cat"]) for b in batch], dim=0)
    conts = torch.stack([(b["cont"]) for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"cat": cats, "cont": conts, "label": labels}


class FinetuneCollator(BaseTabCollator):
    """Simple collator for finetuning - no masking."""
    
    @torch.no_grad()
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        cats = torch.stack([b["cat"] for b in batch], 0)  # (B, L, C)
        conts = torch.stack([b["cont"] for b in batch], 0)  # (B, L, F)
        labels = torch.stack([b["label"] for b in batch], 0)  # (B,)
        
        return {
            "cat": cats,                    # (B, L, C)
            "cont": conts,                  # (B, L, F)
            "downstream_label": labels,     # (B,) - fraud labels
        }