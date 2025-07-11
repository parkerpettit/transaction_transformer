#!/usr/bin/env python
"""
Pre-train TransactionModel to predict the next transaction
(categorical fields + continuous amount).  No fraud head.
Saves encoder weights to  pretrained_backbone.pt
"""
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config            import (ModelConfig, FieldTransformerConfig,
                                SequenceTransformerConfig)
from data.dataset      import TxnDataset, collate_fn, slice_batch
from data.preprocessing import preprocess
from model import TransactionModel
from evaluate          import evaluate            # per-feature val metrics

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
cli = ap.parse_args()

# --- 2. merge file + CLI ---------------
file_params = load_cfg(cli.config)
args = merge(cli, file_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data (load cache or create) ───────────────────────────────────────────
cache = Path(args.data_dir) / "processed_data.pt"
if cache.exists():
    train_df, val_df, enc, cat_features, cont_features = torch.load(cache)
else:
    raw = Path(args.data_dir) / "card_transaction.v1.csv"
    train_df, val_df, enc, cat_features, cont_features = preprocess(raw, args.cat_features, args.cont_features)
    torch.save((train_df, val_df, enc, cat_features, cont_features), cache)

dl_train = DataLoader(
    TxnDataset(train_df, cat_features[0], cat_features, cont_features,
               args.window, args.window),
    batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
dl_val   = DataLoader(
    TxnDataset(val_df, cat_features[0], cat_features, cont_features,
               args.window, args.window),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# ─── Model config with *no* LSTM head ──────────────────────────────────────
cfg = ModelConfig(
    cat_vocab_sizes = {k: len(enc[k]["inv"]) for k in cat_features},
    cont_features   = cont_features,
    emb_dim         = 48,
    dropout         = 0.10,
    padding_idx     = 0,
    total_epochs    = args.epochs,
    field_transformer = FieldTransformerConfig(48, 4, 1, 2, 0.10, 1e-6, True),
    sequence_transformer = SequenceTransformerConfig(256, 4, 4, 2, 0.10, 1e-6, True),
    lstm_config     = None,               # ← key: pre-train only
)
model = TransactionModel(cfg).to(device)

crit_cat  = nn.CrossEntropyLoss()
crit_cont = nn.MSELoss()
optim     = torch.optim.Adam(model.parameters(), lr=args.lr)
vocab_sizes = [len(enc[f]["inv"]) for f in cat_features]

# ─── Training loop ─────────────────────────────────────────────────────────
for ep in range(args.epochs):
    model.train(); tot_loss = 0; nsamp = 0; t0 = time.perf_counter()
    for batch in dl_train:
        ic, xc, m, tc, td = (t.to(device) for t in slice_batch(batch))
        out_cat, out_cont = model(ic, xc, m.bool(), mode="ar")

        # categorical losses field-wise
        start = 0; loss_cat = 0
        for i, V in enumerate(vocab_sizes):
            loss_cat += crit_cat(out_cat[:, start:start+V], tc[:, i])
            start += V
        loss = loss_cat + crit_cont(out_cont, td)

        optim.zero_grad(); loss.backward(); optim.step()
        bs = ic.size(0); tot_loss += loss.item() * bs; nsamp += bs

    train_loss = tot_loss / nsamp
    val_loss, feat_acc = evaluate(model, dl_val, cat_features,
                                  {f: len(enc[f]["inv"]) for f in cat_features},
                                  crit_cat, crit_cont, device)
    print(f"Epoch {ep+1:02}/{args.epochs} "
          f"| train {train_loss:.4f}  val {val_loss:.4f} "
          f"| Δt {time.perf_counter()-t0:.1f}s")

torch.save(model.state_dict(), Path(args.data_dir) / "pretrained_backbone.pt")
print("Pre-training complete ✓  backbone saved.")
