#!/usr/bin/env python
"""
Fine-tune TransactionModel for binary fraud detection.
Loads encoder weights from pretrained_backbone.pt and freezes them unless
--unfreeze is passed.
"""
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config            import (ModelConfig, TransformerConfig, LSTMConfig)
from data.dataset      import TxnDataset, collate_fn, slice_batch
from data.preprocessing import preprocess
from model import TransactionModel
from evaluate   import evaluate, evaluate_binary      # loss + acc per class
from utils             import save_ckpt, load_ckpt

import yaml
from utils import load_cfg, merge

# --- 1. include --config flag -----------
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/pretrain.yaml")
ap.add_argument("--data_dir")         # still allow overrides
ap.add_argument("--epochs",    type=int)
ap.add_argument("--batch_size",type=int)
ap.add_argument("--lr",        type=float)
ap.add_argument("--window",    type=int)
ap.add_argument("--unfreeze",  action="store_true")   # finetune only
ap.add_argument("--cat_features",  nargs="+", default=None,
                help="List of categorical column names")
ap.add_argument("--cont_features", nargs="+", default=None,
                help="List of continuous column names")
ap.add_argument("--resume",      action="store_true")

# ── LSTM head ───────────────────────────────────────────────────────────────
ap.add_argument("--lstm_hidden",  type=int,   help="LSTM hidden size")
ap.add_argument("--lstm_layers",  type=int,   help="Number of LSTM layers")
ap.add_argument("--lstm_classes", type=int, help="Number of LSTM classes")
ap.add_argument("--lstm_dropout", type=float, help="Dropout within LSTM")
cli = ap.parse_args()


# --- 2. merge file + CLI ---------------
file_params = load_cfg(cli.config)
args = merge(cli, file_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data ───────────────────────────────────────────────────────────────────
cache = Path(args.data_dir) / "processed_data.pt"
if cache.exists():
    print("Processed data exists, loading now.")
    train_df, val_df, enc, cat_features, cont_features = torch.load(cache,  weights_only=False)
    print("Processed data loaded.")
else:
    print("Preprocessed data not found. Processing now.")
    raw = Path(args.data_dir) / "card_transaction.v1.csv"
    train_df, val_df, test_df, enc, cat_features, cont_features, scaler = preprocess(raw, args.cat_features, args.cont_features)
    print("Finished processing data. Now saving.")
    torch.save((train_df, val_df, enc, cat_features, cont_features), cache)
    print("Processed data saved.")
print("Creating training loader")
train_loader = DataLoader(
    TxnDataset(train_df, cat_features[0], cat_features, cont_features,
               args.window, args.window),
    batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

print("Creating validation loader")
val_loader   = DataLoader(
    TxnDataset(val_df, cat_features[0], cat_features, cont_features,
               args.window, args.window),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)



# ─── Load backbone & optionally freeze encoder ─────────────────────────────
backbone = Path(args.data_dir) / "pretrained_backbone.pt"
if backbone.exists():
    model.load_state_dict(torch.load(backbone, map_location=device,  weights_only=False), strict=False)
    print(f"Loaded backbone from {backbone}")
else:
    print("⚠  Backbone not found – training from scratch.")

if not args.unfreeze:
    for n, p in model.named_parameters():
        if not n.startswith(("lstm_head", "ar_cat_head", "ar_cont_head")):
            p.requires_grad = False
    print("Encoder frozen.  Pass --unfreeze to fine-tune it.")


if backbone.exists():
    ckpt = torch.load(backbone, map_location=device, weights_only=False)
    finetune_config = ckpt["config"].copy()
    finetune_config.lstm_config = LSTMConfig(
        hidden_size=finetune_config.sequence_transformer.d_model, # must match
        num_layers=args.lstm_num_layers,
        num_classes=args.lstm_num_classes,
        dropout=args.lstm_dropout
    )
    model = TransactionModel(finetune_config).to(device)
    model.load_state_dict(tor)


optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=float(args.lr))
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 400.0], device=device))

best_val, start_ep = load_ckpt(Path(args.data_dir) / "finetune.ckpt",
                               device, model, optim, cat_features, cont_features)

# ─── Training loop ─────────────────────────────────────────────────────────
for ep in range(start_ep, args.epochs):
    t0 = time.perf_counter(); model.train(); tot_loss = 0; epoch_sample_count = 0
    for batch in train_loader:
        cat = batch["cat"][:, :-1].to(device)
        con = batch["cont"][:, :-1].to(device)
        pad = batch["pad_mask"][:, :-1].bool().to(device)
        y   = batch["label"].to(device)

        logits = model(cat, con, pad, mode="fraud")
        loss   = criterion(logits, y)

        optim.zero_grad(); loss.backward(); optim.step()
        batch_size = y.size(0); tot_loss += loss.item() * batch_size; epoch_sample_count += batch_size
    train_loss = tot_loss / epoch_sample_count

    val_loss, val_acc, _ = evaluate_binary(model, val_loader, criterion, device)
    print(f"Epoch {ep+1}/{args.epochs} | train {train_loss:.4f} "
          f"| val {val_loss:.4f} ({val_acc*100:.2f}%) "
          f"| Δt {time.perf_counter()-t0:.1f}s")

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        save_ckpt(model, optim, ep, best_val,
                  Path(args.data_dir) / "finetune.ckpt",
                  cat_features, cont_features, cfg)

print("Fine-tune finished ✓  best val_loss", best_val)
