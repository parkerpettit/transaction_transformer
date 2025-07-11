#!/usr/bin/env python
"""
Pre-train TransactionModel to predict the next transaction
(categorical fields + continuous amount).  No fraud head.
Saves encoder weights to  pretrained_backbone.pt
"""
import wandb
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
import time
from tqdm.auto import tqdm  
from utils import save_ckpt
import signal, sys



def graceful_exit(signum=None, frame=None):
    """Save state, finish wandb, and exit immediately."""
    # # 1. Try to save a last-minute checkpoint (optional but useful)
    # if 'model' in globals():
    #     ckpt_int = Path(args.data_dir) / "pretrained_backbone_interrupt.pt"
    #     torch.save(model.state_dict(), ckpt_int)
    #     print(f"\n[Ctrl-C] Saved interrupt checkpoint → {ckpt_int}")

    # 2. Let W&B know the run is over
    try:
        wandb.finish()
    except Exception:
        pass

    print("[Ctrl-C] Exiting now.")
    sys.exit(0)

# Register SIGINT (Ctrl-C) handler *before* anything long-running starts
signal.signal(signal.SIGINT, graceful_exit)


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
ap.add_argument("--resume", action="store_true",
                help="Resume training from pretrained_backbone.pt in data_dir")

cli = ap.parse_args()

# --- 2. merge file + CLI ---------------
file_params = load_cfg(cli.config)
args = merge(cli, file_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ─── Data (load cache or create) ───────────────────────────────────────────
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
print("Initializing model")
model = TransactionModel(cfg).to(device)
wandb.watch(model, log="gradients", log_freq=100)

crit_cat  = nn.CrossEntropyLoss()
crit_cont = nn.MSELoss()
optim     = torch.optim.Adam(model.parameters(), lr=args.lr)
vocab_sizes = [len(enc[f]["inv"]) for f in cat_features]
print("Starting training loop")

bar_fmt = (
    "{l_bar}{bar:25}| "         # visual bar
    "{n_fmt}/{total_fmt} batches "  # absolute progress
    "({percentage:3.0f}%) | "   # %
    "elapsed: {elapsed} | ETA: {remaining} | "  # timing
    "{rate_fmt} | "             # batches / sec
    "{postfix}"                 # losses go here
)


start_epoch = 0
best_val    = float("inf")

ckpt_path = (Path(args.data_dir) / "pretrained_backbone.pt")
if args.resume and ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 1. weights
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1   # continue with next epoch
        best_val    = ckpt["best_val"]
        print(f"✓ Resumed from {ckpt_path} (epoch {start_epoch}, "
              f"best val {best_val:.4f})")
    else:                                 # weights-only fallback
        model.load_state_dict(ckpt)
        print(f"✓ Loaded pretrained weights from {ckpt_path}")

# ─── wandb must know we’re resuming ───────────────────────────────────────
run = wandb.init(
    project="txn-transformer",
    name   = f"pretrain-{Path(args.data_dir).stem}",
    config = vars(args),
    resume = "allow" if args.resume else False,            # <-- key line
)


# ─── Training loop ─────────────────────────────────────────────────────────
best_val = float("inf")
try:
    for ep in range(start_epoch, args.epochs):
        prog_bar = tqdm(
            train_loader,
            desc=f"Epoch {ep+1}/{args.epochs}",
            unit="batch",
            total=len(train_loader),
            bar_format=bar_fmt,
            ncols=240,               # wider for readability (optional)
            leave=False,             # clear at epoch end
        )
        
        model.train(); tot_loss = 0; epoch_sample_count = 0; t0 = time.perf_counter()
        
        for batch in prog_bar:
        
            cat_input, cont_inp, pad_mask, cat_tgt, cont_tgt = (t.to(device) for t in slice_batch(batch))
            cat_logits, cont_pred = model(cat_input, cont_inp, pad_mask.bool(), mode="ar")

            # categorical losses field-wise
            start = 0
            loss_cat = 0
            for i, vocab_len in enumerate(vocab_sizes):
                loss_cat += crit_cat(cat_logits[:, start:start+vocab_len], cat_tgt[:, i])
                start += vocab_len
            loss_cont = crit_cont(cont_pred, cont_tgt)
            loss_cat /= len(vocab_sizes)
            loss = loss_cat + loss_cont

            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_size = cat_input.size(0)
            tot_loss += loss.item() * batch_size
            epoch_sample_count += batch_size

            prog_bar.set_postfix({
                "tot":  f"{loss.item():.4f}",
                "cat":  f"{loss_cat.item():.4f}", # type: ignore
                "cont": f"{loss_cont.item():.4f}",
            })
        train_loss = tot_loss / epoch_sample_count
        wandb.log({"train/loss": train_loss}, step=ep)

        val_loss, feat_acc = evaluate(model, val_loader, cat_features,
                                    {f: len(enc[f]["inv"]) for f in cat_features},
                                    crit_cat, crit_cont, device)
        wandb.log({
        "val/loss":  val_loss,
        **{f"val/acc_{k}": v for k, v in feat_acc.items()}
        }, step=ep)
        print(f"Epoch {ep+1:02}/{args.epochs} "
            f"| train {train_loss:.4f}  val {val_loss:.4f} "
            f"| Δt {time.perf_counter()-t0:.1f}s")
        if val_loss < best_val - 1e-5:   
            print("New validation loss better than previous. Saving checkpoint.")             
            best_val = val_loss                       
            ckpt_path = Path(args.data_dir) / "pretrained_backbone.pt"
            save_ckpt(                               
                model, optim, ep, best_val,
                ckpt_path, cat_features, cont_features, cfg
            )
            wandb.run.summary["best_val_loss"] = best_val # type: ignore
            print(f"New best ({best_val:.4f}) – checkpoint saved.")
except KeyboardInterrupt:
    graceful_exit()
        
finally:
            
    ckpt_path = Path(args.data_dir) / "pretrained_backbone.pt"
    torch.save(model.state_dict(), ckpt_path)

    artifact = wandb.Artifact("backbone", type="model")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    run.finish()   
    print("Pre-training complete ✓  backbone saved.")
