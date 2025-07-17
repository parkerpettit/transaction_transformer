#!/usr/bin/env python
"""
Fine-tune TransactionModel for binary fraud detection.
Loads encoder weights from pretrained_backbone.pt and freezes them unless
--unfreeze is passed.
"""
import argparse, time, torch, torch.nn as nn
from turtle import back
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from config            import (ModelConfig, TransformerConfig, LSTMConfig)
from data.dataset      import TxnDataset, collate_fn, slice_batch
# from data.preprocessing import preprocess
from model import TransactionModel
from evaluate   import evaluate, evaluate_binary      # loss + acc per class
from utils             import save_ckpt, load_ckpt
import time
import sys
import traceback
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm
from sklearn.utils import resample
import numpy as np
import yaml
from utils import load_cfg, merge

# --- 1. include --config flag -----------
ap = argparse.ArgumentParser(description="Train / fine-tune TransactionModel")

# ───────────── paths / run control ───────────────────────────────────────────
ap.add_argument("--resume",        action="store_true", help="Resume from latest checkpoint in data_dir")
ap.add_argument("--config",        type=str,            help="YAML file with default hyper-params", default="configs/finetune.yaml")
ap.add_argument("--data_dir",      type=str,            help="Root directory of raw or processed data")

# ───────────── training loop hyper-params ────────────────────────────────────
ap.add_argument("--total_epochs",  type=int,            help="Number of training epochs")
ap.add_argument("--batch_size",    type=int,            help="Batch size for training")
ap.add_argument("--lr",            type=float,          help="Initial learning rate")
ap.add_argument("--window",        type=int,            help="Sequence length (transactions per sample)")
ap.add_argument("--stride",        type=int,            help="Stride length between windows")

# ───────────── finetuning control ───────────────────────────────────────────
ap.add_argument("--unfreeze",      action="store_true", help="Unfreeze backbone model parameters for fine-tuning")

# ───────────── feature lists ────────────────────────────────────────────────
ap.add_argument("--cat_features",  type=str,            help="Categorical column names (override YAML)", nargs="+")
ap.add_argument("--cont_features", type=str,            help="Continuous column names  (override YAML)", nargs="+")

# ───────────── LSTM head ─────────────────────────────────────────────────────
ap.add_argument("--lstm_hidden",   type=int,            help="LSTM hidden size")
ap.add_argument("--lstm_layers",   type=int,            help="Number of LSTM layers")
ap.add_argument("--lstm_classes",  type=int,            help="Number of LSTM classes")
ap.add_argument("--lstm_dropout",  type=float,          help="Dropout within LSTM")

cli = ap.parse_args()



# --- 2. merge file + CLI ---------------
file_params = load_cfg(cli.config)
args = merge(cli, file_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data ───────────────────────────────────────────────────────────────────
cache = Path(args.data_dir) / "full_processed.pt"
if cache.exists():
    print("Processed data exists, loading now.")
    train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load(cache,  weights_only=False)
    print("Processed data loaded.")
else:
    print("Preprocessed data not found.")
#     raw = Path(args.data_dir) / "card_transaction.v1.csv"
#     train_df, val_df, test_df, enc, cat_features, cont_features, scaler = preprocess(raw, args.cat_features, args.cont_features)
#     print("Finished processing data. Now saving.")
#     torch.save((train_df, val_df, test_df, enc, cat_features, cont_features, scaler), cache)
#     print("Processed data saved.")

train_ds = TxnDataset(train_df, cat_features[0], cat_features, cont_features,
               args.window, args.window)
val_ds = TxnDataset(val_df, cat_features[0], cat_features, cont_features,
               args.window, args.window)


# from collections import Counter

# def count_txn_labels_direct(dataset):
#     counts = Counter()
#     for i in range(len(dataset)):
#         label = int(dataset[i]["label"].item())
#         counts[label] += 1
#     print(f"non fraud: {counts.get(0,0):,d}")
#     print(f"fraud:     {counts.get(1,0):,d}")
#     return counts

# print(count_txn_labels_direct(train_ds))
# print(count_txn_labels_direct(val_ds))


from collections import Counter
from torch.utils.data import DataLoader

# def count_txn_labels(dataset, batch_size=1024, num_workers=4):
#     """
#     Iterate through TxnDataset (or via a DataLoader) and count how many
#     windows are labeled fraud (1) vs non fraud (0).
#     """
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn  # or your custom collate_fn
#     )
#     counts = Counter()
#     for batch in loader:
#         # if you used your collate_fn, batch["label"] will be a tensor
#         labels = batch["label"].flatten().tolist()
#         counts.update(labels)
#     print(f"non fraud: {counts.get(0,0):,d}")
#     print(f"fraud:     {counts.get(1,0):,d}")
#     return counts
# print("dataloader method counting")
# print(count_txn_labels(train_ds))
# print(count_txn_labels(val_ds))

# Example usage:
# dataset = TxnDataset(df, "user_id", cat_feats, cont_feats, window=20, stride=5)
# count_txn_labels(dataset)

print("Creating training loader")

# def ratio_sampler(ds: TxnDataset, target_pos_frac: float) -> WeightedRandomSampler:
#     """
#     Build a sampler that draws approximately `target_pos_frac` of its samples
#     from the positive class and (1-target_pos_frac) from negative.
#     """
#     # 1) get the per-window “last-transaction” labels
#     gidx = ds.indices[:, 0]
#     offs = ds.indices[:, 1]
#     ends = np.array([
#         ds.group_offsets[g][0] + off + ds.window - 1
#         for g, off in zip(gidx, offs)
#     ], dtype=np.int32)
#     labels = ds.labels[ends]  # array of 0/1

#     # 2) base inverse-frequency weights (makes 50/50 by default)
#     counts = np.bincount(labels, minlength=2)     # [neg_count, pos_count]
#     base_weights = 1.0 / counts[labels]

#     # 3) scale each class so its expected share becomes target_pos_frac
#     #    originally each class has sum(base_weights)==1 => 50/50
#     factor_pos = 2 * target_pos_frac               # e.g. 0.3→0.6
#     factor_neg = 2 * (1 - target_pos_frac)         # e.g. 0.7→1.4

#     # 4) apply the scaling
#     weights = base_weights * np.where(labels==1, factor_pos, factor_neg)

#     return WeightedRandomSampler(
#         weights     = weights, # type: ignore
#         num_samples = len(labels),
#         replacement = True,
#     )
# train_sampler = ratio_sampler(train_ds, target_pos_frac=0.3)
# val_sampler   = ratio_sampler(val_ds,   target_pos_frac=0.3)      # now uses val label distribution

train_loader = DataLoader(
    train_ds,
    batch_size = args.batch_size,
    shuffle=True,
    collate_fn = collate_fn,
)

val_loader = DataLoader(
    val_ds,
    batch_size = args.batch_size,
    shuffle=False,
    collate_fn = collate_fn,
)


# val_loader = DataLoader(
#     val_ds,
#     batch_size   = args.batch_size,
#     shuffle      = False,       # natural ordering
#     collate_fn   = collate_fn,
# )

# print("Creating validation loader")




print("Starting fine-tuning loop")

# progress bar format (reuse from pretrain)
bar_fmt = (
    "{l_bar}{bar:25}| "
    "{n_fmt}/{total_fmt} batches "
    "({percentage:3.0f}%) | "
    "elapsed: {elapsed} | ETA: {remaining} | "
    "{rate_fmt} | "
    "{postfix}"
)

backbone_path = Path(args.data_dir) / "legit_backbone.pt"
finetune_ckpt = Path(args.data_dir) / "finetune.ckpt"
# pos_weight = torch.tensor(818.5349, dtype=torch.float32)

if args.resume and finetune_ckpt.exists():
    # resume entire fine-tune run
    model, best_val, start_epoch = load_ckpt(finetune_ckpt)
    print(f"Resumed fine-tune from epoch {start_epoch}, best val={best_val:.4f}")
else:
    # --- load backbone config & weights ---
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found at {backbone_path}")
    model, _, _ = load_ckpt(backbone_path)
    cfg = model.cfg

    # inject the new LSTM head config
    cfg.lstm_config = LSTMConfig(
        hidden_size = args.lstm_hidden,
        num_layers  = args.lstm_layers,
        num_classes = args.lstm_classes,
        dropout     = args.lstm_dropout,
    )

    model = TransactionModel(cfg)
    # print(cfg)
    # print("------------------")
    # print(cfg.lstm_config)
    # build model and load backbone weights (all except LSTM head)
    print(f"Loaded backbone from {backbone_path}")

    # --- freeze or unfreeze ---
    if not args.unfreeze:
        for name, param in model.named_parameters():
            print(name)
            if not name.startswith("lstm_head"):
                param.requires_grad = False
        print("Transformers frozen. Pass --unfreeze to fine-tune it.")
    else:
        print("Unfreezing entire model for full fine-tuning.")

    # optimizer: only parameters with requires_grad=True
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr
    )

    # classification loss for LSTM head
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_val    = float("inf")

model.to(device)

print("Starting fine-tune training loop")

# Ensure epochs sanity
if start_epoch >= args.total_epochs:
    raise IndexError(
        f"Start epoch ({start_epoch}) >= total_epochs ({args.total_epochs})"
    )

# Initialize Weights & Biases
run = wandb.init(
    project="txn",
    name   = f"finetune-{Path(args.data_dir).stem}",
    config = vars(args),
    resume = "allow" if args.resume else False,
)
wandb.watch(model, log="parameters", log_freq=1000)

# Early-stopping setup
patience = 3
ep_no_improve = 0
try:
    for ep in range(start_epoch, args.total_epochs):
        prog_bar = tqdm(
            train_loader,
            desc=f"Epoch {ep+1}/{args.total_epochs}",
            unit="batch",
            total=len(train_loader),
            bar_format=bar_fmt,
            ncols=200,
            leave=True,
        )

        model.train()
        tot_loss = 0.0
        sample_count = 0
        tot_correct = 0
        t0 = time.perf_counter()
        batch_idx = 0
        for batch in prog_bar:
            cat_inp, cont_inp, cat_tgt, cont_tgt, labels = (t.to(device) for t in slice_batch(batch))
            labels = labels.float()
            fraud_in_batch = labels.sum().item()
            logits = model(cat_inp, cont_inp, mode="fraud")  # (B)

            loss = criterion(logits, labels)
            wandb.log({"training_loss": loss.item()})
            optim.zero_grad()
            loss.backward()
            optim.step()

            probs   = torch.sigmoid(logits)          # (B)
            preds   = (probs > 0.5).float()          # (B)
            tot_correct += (preds == labels).sum().item()
            batch_size = labels.size(0)
            tot_loss += loss.item() * batch_size
            sample_count += batch_size
            prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            batch_idx += 1

        train_loss = tot_loss / sample_count
        train_acc  = tot_correct / sample_count
        # ── validation ──────────────────────────────────────────────────────
        val_loss, val_metrics = evaluate_binary(
            model, val_loader, criterion, device,
            class_names=["non-fraud", "fraud"],
        )

        # log every validation statistic under a common prefix
        # wandb.log({
        #     "val_loss": val_loss,
        #     **{f"val_{k}": v for k, v in val_metrics.items()},
        # }, commit=False)

        epoch_time_min = (time.perf_counter() - t0) / 60.0
        wandb.log({"epoch_time_min": epoch_time_min})

        print(
            f"Epoch {ep+1}/{args.total_epochs} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} "
            f"(acc {val_metrics['accuracy']*100:.2f}%, "
            f"F1 {val_metrics['f1']:.3f}, "
            f"AUC {val_metrics['roc_auc']:.3f}) | "
            f"{epoch_time_min:.2f} min"
        )
        # ── early-stopping / checkpoint ─────────────────────────────────────
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            ep_no_improve = 0
            print("New best validation loss. Saving checkpoint.")
            ckpt_path = Path(args.data_dir) / "finetune.ckpt"
            save_ckpt(
                model, optim, ep, best_val,
                ckpt_path, cat_features, cont_features, cfg
            )
            wandb.log({"best_val_loss": best_val}) # type: ignore
        else:
            ep_no_improve += 1
            if ep_no_improve >= patience:
                print(f"No improvement for {patience} epochs. Stopping early.")
                break

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        run.alert(
            title="CUDA OOM",
            text=f"OOM at batch_size={args.batch_size}. Marking failed."
        )
        run.finish(exit_code=97)
        sys.exit(97)
    else:
        traceback.print_exc()
        run.finish(exit_code=98)
        sys.exit(98)

finally:
    ckpt_path = Path(args.data_dir) / "finetune.ckpt"
    if wandb.run is not None and ckpt_path.exists():
        artifact = wandb.Artifact("finetune-model", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        run.finish()
    print(f"Fine-tuning complete. Best val_loss: {best_val:.4f}")