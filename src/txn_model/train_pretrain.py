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
from utils import load_cfg, merge, load_ckpt
import time
from tqdm.auto import tqdm  
from utils import save_ckpt
import signal, sys

def show_samples(cat_inp, cat_tgt, cat_preds,
                 cont_tgt, cont_preds,
                 enc, cat_features, cont_features,
                 n=3):
    """
    cat_inp   : [B, W, F_cat]  (not used here but available if you want)
    cat_tgt   : [B, F_cat]     codes of last timestep
    cat_preds : [B, F_cat]     argmax codes
    cont_tgt  : [B, F_cont]
    cont_preds: [B, F_cont]
    """
    B = cat_tgt.size(0)
    n = min(n, B)

    # decode function
    def d(code, feat_name):
        inv = enc[feat_name]["inv"]
        return inv[code] if code < len(inv) else f"<UNK:{code}>"

    for b in range(n):
        print(f"\n─ Sample {b} ─")
        for i, feat in enumerate(cat_features):
            tgt_code  = cat_tgt[b, i].item()
            pred_code = cat_preds[b, i].item()
            print(f"{feat:<18}: tgt={d(tgt_code, feat)} | pred={d(pred_code, feat)}")
        for j, feat in enumerate(cont_features):
            t_val = cont_tgt[b, j].item()
            p_val = cont_preds[b, j].item()
            print(f"{feat:<18}: tgt={t_val:>8.2f} | pred={p_val:>8.2f}")
        print("─" * 40)

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
if args.resume:
    best_val, start_epoch = load_ckpt(
        path=ckpt_path,
        device=device,
        model=model,
        optimizer=optim,
        cat_features=cat_features,
        cont_features=cont_features,
    )
else:
    best_val, start_epoch = float("inf"), 0

# ─── wandb must know we’re resuming ───────────────────────────────────────
run = wandb.init(
    project="txn-transformer",
    name   = f"pretrain-{Path(args.data_dir).stem}",
    config = vars(args),
    resume = "allow" if args.resume else False,            # <-- key line
)

wandb.watch(model, log="gradients", log_freq=100)

# ─── Training loop ─────────────────────────────────────────────────────────
best_val = float("inf")
patience = 3 # number of acceptable consecutive epochs without validation loss improvement
ep_without_improvement = 0
# 1. Make sure 'User' is in the input
sample = next(iter(train_loader))
print("---------------------------------------------------------")
print(sample["cat"][0, :, 0])       # first feature across the window
# Should show the same user-ID repeated.

# 2. Confirm class count
print(train_df["User"].nunique())   # probably > 1 000

# 3. Random baseline
most_common = train_df["User"].value_counts().iloc[0] / len(train_df)
print("Chance-level (always-pick-top):", most_common)
print('------------------------------------------------------')

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
        batch_idx = 0
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
            ep_without_improvement = 0
            print("New validation loss better than previous. Saving checkpoint.")             
            best_val = val_loss                       
            ckpt_path = Path(args.data_dir) / "pretrained_backbone.pt"
            save_ckpt(                               
                model, optim, ep, best_val,
                ckpt_path, cat_features, cont_features, cfg
            )
            wandb.run.summary["best_val_loss"] = best_val # type: ignore
            print(f"New best ({best_val:.4f}) – checkpoint saved.")
        else:
            if ep_without_improvement >= patience:
                break
            else:
                ep_without_improvement += 1
        # after computing preds for this batch
        if batch_idx == 0:   # print only from the first batch to avoid spam
            # gather predictions in the same per-feature layout you used for accuracy
            # (re-run the start/end loop or cache the per-feature argmax)
            start = 0
            preds_cat = torch.empty_like(cat_tgt)
            for i, V in enumerate(vocab_sizes):
                end = start + V
                preds_cat[:, i] = cat_logits[:, start:end].argmax(dim=1)
                start = end

            show_samples(
                cat_input, cat_tgt, preds_cat,
                cont_tgt,  cont_pred,
                enc, cat_features, cont_features,
                n=3
            )
        batch_idx += 1

                
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
