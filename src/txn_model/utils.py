import os
import time
import torch
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, best_val, path, cat_features, cont_features, config):
    torch.save({
        "epoch":       epoch,
        "best_val":    best_val,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "cat_features": cat_features,
        "cont_features": cont_features,
        "config": config
    }, path)

    # ── verification ──
    cp = torch.load(path, map_location="cpu")
    assert cp["epoch"] == epoch,  "checkpoint epoch mismatch – file not overwritten?"

    mod_time = time.ctime(os.path.getmtime(path))
    size_mb  = os.path.getsize(path) / 1_048_576
    print(f"Checkpoint for epoch {epoch} written ({size_mb:.1f} MB, {mod_time})")


def load_or_initialize_checkpoint(
    base_path: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cat_features: list[str],
    cont_features: list[str]
) -> tuple[float, int]:
    """
    Load checkpoint if it exists and matches the feature lists; otherwise rename
    the old checkpoint and start fresh.

    Args:
        base_path: Path to the checkpoint file.
        device: Torch device for loading.
        model: The model to load state into.
        optimizer: The optimizer to load state into.
        cat_features: Current list of categorical feature names.
        cont_features: Current list of continuous feature names.

    Returns:
        best_val: Best validation loss (infinite if starting fresh).
        start_epoch: Epoch number to start training from (1-based).
    """
    if os.path.exists(base_path):
        ckpt = torch.load(base_path, map_location=device, weights_only=False)
        old_cat = ckpt.get("cat_features", [])
        old_cont = ckpt.get("cont_features", [])
        mismatch_cat = old_cat != cat_features
        mismatch_cont = old_cont != cont_features

        if mismatch_cat or mismatch_cont:
            # Report mismatches
            print("Feature mismatch detected in checkpoint:")
            if mismatch_cat:
                print("   Categorical features differ:")
                print(f"      saved: {old_cat}")
                print(f"      now:   {cat_features}")
            if mismatch_cont:
                print("   Continuous features differ:")
                print(f"      saved: {old_cont}")
                print(f"      now:   {cont_features}")

            # Rename old checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feat_sig = f"C{len(old_cat)}_Ct{len(old_cont)}"
            new_name = f"txn_old_{feat_sig}_{timestamp}.pt"
            os.rename(base_path, new_name)
            print(f"Renamed old checkpoint to: {new_name}\n")

            best_val = float("inf")
            start_epoch = 1
        else:
            # Safe to resume
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            best_val = ckpt.get("best_val", float("inf"))
            start_epoch = ckpt.get("epoch", 0) + 1
    else:
        best_val = float("inf")
        start_epoch = 0

    return best_val, start_epoch

import torch
import numpy as np

def extract_latents_per_card(df_split, model, cat_cols, cont_cols, device):
    """
    One‐pass per‐card latent extraction for a TransactionModel.

    df_split : DataFrame (train_df / val_df / test_df), with encoded & scaled features
    model    : your TransactionModel instance
    cat_cols : list of categorical column names
    cont_cols: list of continuous column names
    device   : torch.device('cuda') or ('cpu')

    Returns:
      X (N×D) latent matrix and y (N,) label vector
    """
    model.eval()
    X_parts, y_parts = [], []

    with torch.no_grad():
        for cc, g in df_split.groupby("cc_num", sort=False):
            # Build tensors of shape (1, L, C) and (1, L, F)
            L = len(g)
            cat_tensor  = torch.tensor(
                g[cat_cols].values, dtype=torch.long, device=device
            ).unsqueeze(0)                 # (1, L, C)
            cont_tensor = torch.tensor(
                g[cont_cols].values, dtype=torch.float32, device=device
            ).unsqueeze(0)                 # (1, L, F)
            pad_mask = torch.zeros(
                (1, L), dtype=torch.bool, device=device
            )                               # no padding in full-history mode

            # 1) Embed
            embeddings = model.embedder(cat_tensor, cont_tensor)  # (1, L, E)

            # 2) Field Transformer + flatten
            ft = model.field_transformer(embeddings)              # (1, L, k, d)
            intra = ft.flatten(start_dim=1)                       # (1, L*k*d)

            # 3) Row projection back to sequence shape
            row_repr = model.row_projector(intra)                 # (1, L*m_flat)
            row_repr = row_repr.view(1, L, -1)                    # (1, L, m)

            # 4) Sequence Transformer
            seq_out = model.sequence_transformer(
                row_repr, padding_mask=pad_mask
            )                                                     # (1, L, m)

            # 5) Project *every* position to latent space
            seq_flat   = seq_out.view(L, -1)                      # (L, m)
            latent_all = model.output_projector(seq_flat)         # (L, k*d)

            # 6) Collect
            X_parts.append(latent_all.cpu().numpy())              # list of (L, D)
            y_parts.append(g["is_fraud"].to_numpy())              # list of (L,)

    # Concatenate all cards
    X = np.vstack(X_parts)                                     # (ΣL, D)
    y = np.concatenate(y_parts)                                # (ΣL,)
    return X, y




    

