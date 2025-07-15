#!/usr/bin/env python
"""
train_lightgbm.py

End-to-end pipeline to:
  1) Load or preprocess transaction data
  2) Extract Transformer embeddings (last-token) via DataLoader → NumPy memmaps
  3) Train a LightGBM classifier on those embeddings (no separate .bin step)
  4) Evaluate on train/val/test with full metrics & log everything to Weights & Biases

Usage:
  python train_lightgbm.py \
    --data_dir path/to/data \
    --model_ckpt path/to/pretrained_backbone.pt \
    --wandb_project my_wandb_proj \
    --rebuild_embeddings

If the files train_emb.npy, train_lbl.npy, etc. already exist under --data_dir/features/,
they’ll be loaded directly.  Otherwise they’ll be built from scratch.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import lightgbm as lgb
import wandb
from numpy.lib.format import open_memmap
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    f1_score, confusion_matrix,
    classification_report,
)

# your project’s imports:
from data.preprocessing import preprocess
from data.dataset import TxnDataset, collate_fn
from utils import load_ckpt


def dump_embeddings(loader, model, device, split, features_dir, rebuild=False):
    """
    Dump last-token embeddings + labels to memmap .npy files under features_dir.
    """
    emb_path = features_dir / f"{split}_emb.npy"
    lbl_path = features_dir / f"{split}_lbl.npy"
    if emb_path.exists() and lbl_path.exists() and not rebuild:
        print(f"[{split}] embeddings already exist, skipping dump.")
        return emb_path, lbl_path

    print(f"[{split}] dumping embeddings → {emb_path}, labels → {lbl_path} ...")
    n = len(loader.dataset)
    # infer feature dimension
    batch = next(iter(loader))
    with torch.no_grad():
        M = model(
            batch["cat"].to(device),
            batch["cont"].to(device),
            batch["pad_mask"].to(device),
            mode="lightgbm"
        ).shape[-1]

    # create memmaps
    embs = open_memmap(str(emb_path), dtype="float32", mode="w+",
                       shape=(n, M))
    lbls = open_memmap(str(lbl_path), dtype="int8", mode="w+",
                       shape=(n,))

    model.eval()
    idx = 0
    with torch.no_grad():
        for batch in loader:
            B = batch["cat"].shape[0]
            feats = model(
                batch["cat"].to(device),
                batch["cont"].to(device),
                batch["pad_mask"].to(device),
                mode="lightgbm"
            ).cpu().numpy()
            embs[idx:idx+B] = feats
            lbls[idx:idx+B] = batch["label"].numpy()
            idx += B

    print(f"[{split}] done (n={n}, M={M}).")
    return emb_path, lbl_path


def evaluate_and_log(split, y_true, y_proba, class_names):
    """
    Compute metrics for one split and log to wandb.
    """
    y_pred = (y_proba >= 0.5).astype(int)
    wandb.log({
        f"{split}/auc":       roc_auc_score(y_true, y_proba),
        f"{split}/accuracy":  accuracy_score(y_true, y_pred),
        f"{split}/precision": precision_score(y_true, y_pred),
        f"{split}/recall":    recall_score(y_true, y_pred),
        f"{split}/f1":        f1_score(y_true, y_pred),
    })
    wandb.log({
        f"{split}/confusion_matrix":
            wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
    })


def main():
    # load wandb config
    run = wandb.init()
    cfg = run.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ───── 1) Load or preprocess raw DataFrames ───────────────────────────
    cache = Path(cfg.data_dir) / "processed_data.pt"
    if cache.exists():
        print("Loading processed_data.pt ...")
        train_df, val_df, test_df, enc, cat_feats, cont_feats, scaler = torch.load(cache)
    else:
        print("Preprocessing raw CSV ...")
        raw = Path(cfg.data_dir) / "card_transaction.v1.csv"
        train_df, val_df, test_df, enc, cat_feats, cont_feats, scaler = preprocess(
            raw, cfg.cat_features, cfg.cont_features
        )
        torch.save((train_df, val_df, test_df, enc, cat_feats, cont_feats, scaler), cache)

    # ───── 2) Build DataLoaders ────────────────────────────────────────────
    features_dir = Path(cfg.data_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        TxnDataset(train_df, cat_feats[0], cat_feats, cont_feats, cfg.window, cfg.stride),
        batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TxnDataset(val_df, cat_feats[0], cat_feats, cont_feats, cfg.window, cfg.stride),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        TxnDataset(test_df, cat_feats[0], cat_feats, cont_feats, cfg.window, cfg.stride),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    # ───── 3) Load pretrained backbone ────────────────────────────────────
    backbone, _, _ = load_ckpt(cfg.model_ckpt)
    model = backbone.to(device)

    # ───── 4) Dump / load embeddings & labels ──────────────────────────────
    # splits = train, val, test
    for split, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        dump_embeddings(loader, model, device, split, features_dir, rebuild=cfg.rebuild_embeddings)

    X_train = np.load(features_dir / "train_emb.npy", mmap_mode="r")
    y_train = np.load(features_dir / "train_lbl.npy", mmap_mode="r")
    X_val   = np.load(features_dir / "val_emb.npy",   mmap_mode="r")
    y_val   = np.load(features_dir / "val_lbl.npy",   mmap_mode="r")
    X_test  = np.load(features_dir / "test_emb.npy",  mmap_mode="r")
    y_test  = np.load(features_dir / "test_lbl.npy",  mmap_mode="r")

    # ───── 5) Build LightGBM Datasets ─────────────────────────────────────
    train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    val_ds   = lgb.Dataset(X_val,   label=y_val,   reference=train_ds)

    # ───── 6) Train LightGBM ───────────────────────────────────────────────
    params = {
        "objective":               cfg.objective,
        "boosting":                cfg.boosting_type,
        "num_leaves":              cfg.num_leaves,
        "max_depth":               cfg.max_depth,
        "learning_rate":           cfg.learning_rate,
        "min_data_in_leaf":        cfg.min_data_in_leaf,
        "min_sum_hessian_in_leaf": cfg.min_sum_hessian_in_leaf,
        "min_gain_to_split":       cfg.min_gain_to_split,
        "lambda_l1":               cfg.lambda_l1,
        "lambda_l2":               cfg.lambda_l2,
        "is_unbalance":            cfg.is_unbalance,
        "scale_pos_weight":        cfg.scale_pos_weight,
        "subsample":               cfg.subsample,
        "subsample_freq":          cfg.subsample_freq,
        "colsample_bytree":        cfg.colsample_bytree,
        "feature_fraction":        cfg.feature_fraction,
        "max_bin":                 cfg.max_bin,
        "metric":                  cfg.metric,
        "verbosity":               -1,
        "seed":                    cfg.seed,
    }

    bst = lgb.train(
        params,
        train_ds,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds),
            lgb.log_evaluation(cfg.verbose)
        ],
    )

    # ───── 7) Evaluate & log on all splits ────────────────────────────────
    evaluate_and_log("train", y_train, bst.predict(X_train), cfg.class_names)
    evaluate_and_log("val",   y_val,   bst.predict(X_val),   cfg.class_names)
    evaluate_and_log("test",  y_test,  bst.predict(X_test),  cfg.class_names)

    # full classification report
    y_test_pred = (bst.predict(X_test) >= 0.5).astype(int)
    run.summary["classification_report"] = classification_report(
        y_test, y_test_pred, target_names=cfg.class_names
    )

    # ───── 8) Save final model ─────────────────────────────────────────────
    model_path = features_dir / "lgbm_best_model.txt"
    bst.save_model(str(model_path))
    print(f"Saved LightGBM model to {model_path}")

    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",            type=str, required=True,
                    help="Root directory of raw/processed data")
    ap.add_argument("--model_ckpt",          type=str, required=True,
                    help="Path to pretrained TransactionModel checkpoint")
    ap.add_argument("--wandb_project",       type=str, default="transaction_transformer",
                    help="Weights & Biases project name")
    ap.add_argument("--wandb_entity",        type=str, default=None,
                    help="W&B entity (team/user)")
    ap.add_argument("--rebuild_embeddings",  action="store_true",
                    help="Force re-extraction of embeddings even if files exist")

    args = ap.parse_args()

    # defaults for both CLI-run and sweeps
    sweep_defaults = {
        # paths & rebuild
        "data_dir":           args.data_dir,
        "model_ckpt":         args.model_ckpt,
        "rebuild_embeddings": args.rebuild_embeddings,
        # data/window settings (must match pretraining)
        "batch_size":  384,
        "window":      10,
        "stride":      5,
        # preprocessing (only used if processed_data.pt is missing)
        "cat_features": ["User","Card","Use Chip","Merchant Name","Merchant City",
                         "Merchant State","Zip","MCC","Errors?","Year","Month","Day","Hour"],
        "cont_features":["Amount"],
        # LightGBM hyperparameters
        "objective":    "binary",
        "boosting_type":"gbdt",
        "num_leaves":   31,
        "max_depth":    -1,
        "learning_rate":0.1,
        "num_boost_round":       1000,
        "early_stopping_rounds": 10,
        "subsample":             1.0,
        "subsample_freq":        0,
        "colsample_bytree":      1.0,
        "feature_fraction":      1.0,
        "min_data_in_leaf":      20,
        "min_sum_hessian_in_leaf":1e-3,
        "min_gain_to_split":     0.0,
        "lambda_l1":             0.0,
        "lambda_l2":             0.0,
        "is_unbalance":          True,
        "scale_pos_weight":      1.0,
        "max_bin":               255,
        "metric":                "auc",
        "verbose":               0,
        "seed":                  42,
        # W&B reporting
        "class_names": ["non-fraud","fraud"],
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=sweep_defaults,
        job_type="train_lightgbm",
        reinit=True
    )
    main()
