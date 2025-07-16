import wandb
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config            import (ModelConfig, TransformerConfig)
from data.dataset      import TxnDataset, collate_fn, slice_batch
from data.preprocessing import preprocess
from model import TransactionModel
from evaluate          import evaluate            # per-feature val metrics
import numpy as np
import yaml
from utils import load_cfg, merge, load_ckpt
import time
from tqdm.auto import tqdm  
from utils import save_ckpt
import signal, sys
import wandb, torch, traceback, sys

import lightgbm as lgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import wandb




ap = argparse.ArgumentParser(description="Train / fine-tune TransactionModel")


# ───────────── paths / run control ───────────────────────────────────────────
ap.add_argument("--resume", action="store_true",    help="Resume from latest checkpoint in data_dir")
ap.add_argument("--config",            type=str,    help="YAML file with default hyper-params", default="configs/pretrain.yaml")
ap.add_argument("--data_dir",          type=str,    help="Root directory of raw or processed data")

# ───────────── training loop hyper-params ────────────────────────────────────
ap.add_argument("--total_epochs",      type=int,    help="Number of training epochs")
ap.add_argument("--batch_size",        type=int,    help="Batch size for training")
ap.add_argument("--lr",                type=float,  help="Initial learning rate")
ap.add_argument("--window",            type=int,    help="Sequence length (transactions per sample)")
ap.add_argument("--stride",            type=int,    help="Stride length between windows")

# ───────────── feature lists ────────────────────────────────────────────────
ap.add_argument("--cat_features",      type=str,    help="Categorical column names (override YAML)", nargs="+")
ap.add_argument("--cont_features",     type=str,    help="Continuous column names  (override YAML)", nargs="+")

# ───────────── architecture: embedding layer ────────────────────────────────
ap.add_argument("--emb_dropout",        type=float, help="Dropout after embedding layer")

# ── Field-level transformer (intra-row) ──────────────────────────────────────
ap.add_argument("--ft_d_model",         type=int,   help="Field-transformer hidden dimension")
ap.add_argument("--ft_depth",           type=int,   help="Field-transformer number of layers")
ap.add_argument("--ft_n_heads",         type=int,   help="Field-transformer number of attention heads")
ap.add_argument("--ft_ffn_mult",        type=int,   help="Field-transformer feedforward expansion factor")
ap.add_argument("--ft_dropout",         type=float, help="Dropout within field-transformer")
ap.add_argument("--ft_layer_norm_eps",  type=float, help="Layer norm epsilon for field-transformer")
ap.add_argument("--ft_norm_first",      type=bool,  help="Norm first for field-transformer")
# ── Sequence-level transformer (inter-row) ───────────────────────────────────
ap.add_argument("--seq_d_model",        type=int,   help="Sequence-transformer hidden dimension")
ap.add_argument("--seq_depth",          type=int,   help="Sequence-transformer number of layers")
ap.add_argument("--seq_n_heads",        type=int,   help="Sequence-transformer number of attention heads")
ap.add_argument("--seq_ffn_mult",       type=int,   help="Sequence-transformer feedforward expansion factor")
ap.add_argument("--seq_dropout",        type=float, help="Dropout within sequence-transformer")
ap.add_argument("--seq_layer_norm_eps", type=float, help="Layer norm epsilon for sequence-transformer")
ap.add_argument("--seq_norm_first",     type=bool,  help="Norm first for sequence-transformer")


# ── Final classification layer ──────────────────────────────────────────────
ap.add_argument("--clf_dropout",  type=float, help="Dropout before final classification layer")


# LightGBM objective & boosting
ap.add_argument("--lgb_objective",             type=str,   default="binary", help="Objective function (e.g., binary, multiclass)")
ap.add_argument("--lgb_boosting_type",         type=str,   default="gbdt",   help="Boosting type: gbdt, dart, goss, rf")

# Core model complexity
ap.add_argument("--lgb_num_leaves",            type=int,   default=31,      help="Max number of leaves per tree")
ap.add_argument("--lgb_max_depth",             type=int,   default=-1,      help="Max tree depth (-1=no limit)")

# Learning rate & iterations
ap.add_argument("--lgb_learning_rate",         type=float, default=0.1,     help="Boosting learning rate")
ap.add_argument("--lgb_n_estimators",          type=int,   default=100,     help="Number of boosting rounds")

# Data sampling (bagging)
ap.add_argument("--lgb_subsample",             type=float, default=1.0,     help="Fraction of data to use per iteration")
ap.add_argument("--lgb_subsample_freq",        type=int,   default=0,       help="Re-sample every k iterations (0=disabled)")

# Feature sampling
ap.add_argument("--lgb_colsample_bytree",      type=float, default=1.0,     help="Fraction of features per tree")
ap.add_argument("--lgb_feature_fraction",      type=float, default=1.0,     help="Alias for colsample_bytree")

# Regularization
ap.add_argument("--lgb_min_data_in_leaf",      type=int,   default=20,      help="Min records per leaf")
ap.add_argument("--lgb_min_sum_hessian_in_leaf", type=float, default=1e-3,   help="Min sum Hessian per leaf")
ap.add_argument("--lgb_min_gain_to_split",     type=float, default=0.0,     help="Min gain required to split")
ap.add_argument("--lgb_lambda_l1",             type=float, default=0.0,     help="L1 regularization")
ap.add_argument("--lgb_lambda_l2",             type=float, default=0.0,     help="L2 regularization")

# Handling class imbalance
ap.add_argument("--lgb_is_unbalance",          action="store_true",         help="Enable built-in unbalanced dataset handling")
ap.add_argument("--lgb_scale_pos_weight",      type=float, default=1.0,     help="Weight for positive class")

# Binning
ap.add_argument("--lgb_max_bin",               type=int,   default=255,     help="Max number of bins for continuous features")

# Early stopping & evaluation
ap.add_argument("--lgb_metric",               type=str,   default="auc",    help="Evaluation metric (e.g., auc, binary_logloss)")
ap.add_argument("--lgb_early_stopping_rounds", type=int,   default=10,      help="Rounds of no improvement before stopping")

# Misc
ap.add_argument("--lgb_random_state",         type=int,   default=42,      help="Random seed")
ap.add_argument("--lgb_verbose",              type=int,   default=2,       help="Verbosity level")


cli = ap.parse_args()



# --- 2. merge file + CLI ---------------
file_params = load_cfg(cli.config)
args = merge(cli, file_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ─── Data (load cache or create) ───────────────────────────────────────────
cache = Path(args.data_dir) / "processed_data.pt"
if cache.exists():
    print("Processed data exists, loading now.")
    train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load(cache,  weights_only=False)
    print("Processed data loaded.")
else:
    print("Preprocessed data not found. Processing now.")
    raw = Path(args.data_dir) / "card_transaction.v1.csv"
    train_df, val_df, test_df, enc, cat_features, cont_features, scaler = preprocess(raw, args.cat_features, args.cont_features)
    print("Finished processing data. Now saving.")
    torch.save((train_df, val_df, test_df, enc, cat_features, cont_features, scaler), cache)
    print("Processed data saved.")


print("Creating training loader")
train_loader = DataLoader(
    TxnDataset(train_df, cat_features[0], cat_features, cont_features,
            args.window, args.stride),
    batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)

print("Creating validation loader")
val_loader   = DataLoader(
    TxnDataset(val_df, cat_features[0], cat_features, cont_features,
            args.window, args.stride),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)


print("Creating validation loader")
test_loader   = DataLoader(
    TxnDataset(test_df, cat_features[0], cat_features, cont_features,
            args.window, args.stride),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

model, _, _ = load_ckpt("data/pretrained_backbone.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# import csv, os

# import numpy as np
# import os

# def dump_memmap(split_name, loader, model, device, out_dir="features"):
#     os.makedirs(out_dir, exist_ok=True)

#     # figure out counts
#     # 1) first pass: count rows
#     n_rows = 0
#     for _ in loader:
#         n_rows += _.get("cat").shape[0]

#     # 2) get feature dim
#     batch = next(iter(loader))
#     with torch.no_grad():
#         M = model(
#             batch["cat"].to(device),
#             batch["cont"].to(device),
#             batch["pad_mask"].to(device),
#             mode="lightgbm"
#         ).shape[-1]

#     # 3) create memmap arrays
#     emb_path = os.path.join(out_dir, f"{split_name}_emb.npy")
#     lbl_path = os.path.join(out_dir, f"{split_name}_lbl.npy")
#     embs = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_rows, M))
#     lbls = np.memmap(lbl_path, dtype="int8",   mode="w+", shape=(n_rows,))

#     # 4) fill them
#     idx = 0
#     with torch.no_grad():
#         for batch in loader:
#             B = batch["cat"].shape[0]
#             feat = model(
#                 batch["cat"].to(device),
#                 batch["cont"].to(device),
#                 batch["pad_mask"].to(device),
#                 mode="lightgbm"
#             ).cpu().numpy()  # (B, M)
#             embs[idx:idx+B] = feat
#             lbls[idx:idx+B] = batch["label"].numpy()
#             idx += B

#     del embs, lbls  # flush to disk
#     return emb_path, lbl_path

# # example:
# # train_emb, train_lbl = dump_memmap("train", train_loader, model, device)
# # val_emb,   val_lbl   = dump_memmap("val",   val_loader,   model, device)
# # test_emb,  test_lbl  = dump_memmap("test",  test_loader,  model, device)
# import numpy as np
# import lightgbm as lgb

# def make_lgb_binary(emb_path, lbl_path, out_bin, n_rows, M):
#     # Open with known shape and dtype
#     X = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_rows, M))
#     y = np.memmap(lbl_path, dtype="int8",   mode="r", shape=(n_rows,))

#     ds = lgb.Dataset(X, label=y, free_raw_data=True)
#     ds.save_binary(out_bin)
#     print("Wrote", out_bin)

# # Usage
# # make_lgb_binary("features/train_emb.npy", "features/train_lbl.npy", "features/train.bin", 3411550, 384)
# # make_lgb_binary("features/val_emb.npy",   "features/val_lbl.npy",   "features/val.bin", 728789, 384 )
# # make_lgb_binary("features/test_emb.npy",  "features/test_lbl.npy",  "features/test.bin", 8431102, 384)
# # 1) Infer number of windows (rows) directly from your dataset
# n_train = len(train_loader.dataset)
# n_val   = len(val_loader.dataset)
# n_test  = len(test_loader.dataset)

# # 2) Infer embedding dimension M with a dummy batch
# batch = next(iter(train_loader))
# with torch.no_grad():
#     emb = model(
#         batch["cat"].to(device),
#         batch["cont"].to(device),
#         batch["pad_mask"].to(device),
#         mode="lightgbm"
#     )
# M = emb.shape[-1]

# print(f"Will write {n_train=} rows, {n_val=}, {n_test=} rows, each with {M=} features")

# # 3) Now call make_lgb_binary without re-dumping
# make_lgb_binary("features/train_emb.npy", "features/train_lbl.npy", "features/train.bin", n_train, M)
# make_lgb_binary("features/val_emb.npy",   "features/val_lbl.npy",   "features/val.bin",   n_val,   M)
# make_lgb_binary("features/test_emb.npy",  "features/test_lbl.npy",  "features/test.bin",  n_test,  M)

# # 1) Init W&B
# run = wandb.init(
#     project="lightgbm",
#     name="test_run",
#     config=vars(args),
#     resume="allow" if args.resume else False,
# )

# # 2) Extract train/val (IN MEMORY—this is the only big allocation)
# X_train, y_train = extract_lightgbm_features(model, train_loader, device)
# X_val,   y_val   = extract_lightgbm_features(model, val_loader,   device)

# # 3) Build LightGBM Datasets and free the raw arrays immediately
# train_data = lgb.Dataset(
#     X_train, y_train,
#     free_raw_data=True   # <<< frees X_train/y_train after binning
# )
# val_data = lgb.Dataset(
#     X_val, y_val,
#     reference=train_data,
#     free_raw_data=True   # <<< frees X_val/y_val after binning
# )

# del X_train, y_train, X_val, y_val
# # at this point your Python process should have released ~2-3 GB of RAM

# # 4) Train with early stopping on val
# params = {
#     "objective":           args.lgb_objective,
#     "boosting":            args.lgb_boosting_type,
#     "num_leaves":          args.lgb_num_leaves,
#     "max_depth":           args.lgb_max_depth,
#     "learning_rate":       args.lgb_learning_rate,
#     "min_data_in_leaf":    args.lgb_min_data_in_leaf,
#     "min_sum_hessian_in_leaf": args.lgb_min_sum_hessian_in_leaf,
#     "min_gain_to_split":   args.lgb_min_gain_to_split,
#     "lambda_l1":           args.lgb_lambda_l1,
#     "lambda_l2":           args.lgb_lambda_l2,
#     "is_unbalance":        args.lgb_is_unbalance,
#     "scale_pos_weight":    args.lgb_scale_pos_weight,
#     "subsample":           args.lgb_subsample,
#     "subsample_freq":      args.lgb_subsample_freq,
#     "colsample_bytree":    args.lgb_colsample_bytree,
#     "feature_fraction":    args.lgb_feature_fraction,
#     "max_bin":             args.lgb_max_bin,
#     "metric":              args.lgb_metric,
#     "verbose":             -1,
#     "seed":                args.lgb_random_state,
# }
# callbacks = [lgb.early_stopping(args.lgb_early_stopping_rounds)]

# bst = lgb.train(
#     params,
#     train_data,
#     num_boost_round=args.lgb_n_estimators,
#     valid_sets=[val_data],
#     callbacks=callbacks,
# )

# # 5) Stream test predictions in batches (never build X_test all at once)
# all_true, all_proba, all_pred = [], [], []
# for batch in test_loader:
#     cat = batch["cat"].to(device)
#     cont = batch["cont"].to(device)
#     pad = batch["pad_mask"].to(device)
#     true = batch["label"].cpu().numpy()
#     emb  = model(cat, cont, pad, mode="lightgbm").cpu().numpy()  # (b, M)
#     proba = bst.predict(emb)                                     # (b,)
#     pred  = (proba >= 0.5).astype(int)
#     all_true.append(true)
#     all_proba.append(proba)
#     all_pred.append(pred)

# y_true = np.concatenate(all_true)
# y_proba = np.concatenate(all_proba)
# y_pred  = np.concatenate(all_pred)

# # 6) Compute metrics and log to W&B
# metrics = {
#     "test/auc":       roc_auc_score(y_true, y_proba),
#     "test/accuracy":  accuracy_score(y_true, y_pred),
#     "test/precision": precision_score(y_true, y_pred),
#     "test/recall":    recall_score(y_true, y_pred),
#     "test/f1":        f1_score(y_true, y_pred),
# }
# # confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# wandb.log({
#     **metrics,
#     "test/confusion_matrix": wandb.plot.confusion_matrix(
#         probs=None,
#         y_true=y_true,
#         preds=y_pred,
#         class_names=["non-fraud", "fraud"]
#     )
# })
# # full textual report
# wandb.run.summary["classification_report"] = classification_report(
#     y_true, y_pred, target_names=["non-fraud","fraud"]
# )

# run.finish()


import lightgbm as lgb

# 1) Load your binary Datasets
train_ds = lgb.Dataset("features/train.bin")
val_ds   = lgb.Dataset("features/val.bin", reference=train_ds)

# 2) Build the params dict from your args
params = {
    "objective":             args.lgb_objective,
    "boosting":              args.lgb_boosting_type,
    "num_leaves":            args.lgb_num_leaves,
    "max_depth":             args.lgb_max_depth,
    "learning_rate":         args.lgb_learning_rate,
    "min_data_in_leaf":      args.lgb_min_data_in_leaf,
    "min_sum_hessian_in_leaf": args.lgb_min_sum_hessian_in_leaf,
    "min_gain_to_split":     args.lgb_min_gain_to_split,
    "lambda_l1":             args.lgb_lambda_l1,
    "lambda_l2":             args.lgb_lambda_l2,
    "is_unbalance":          args.lgb_is_unbalance,
    "scale_pos_weight":      args.lgb_scale_pos_weight,
    "subsample":             args.lgb_subsample,
    "subsample_freq":        args.lgb_subsample_freq,
    "colsample_bytree":      args.lgb_colsample_bytree,
    "feature_fraction":      args.lgb_feature_fraction,
    "max_bin":               args.lgb_max_bin,
    "metric":                args.lgb_metric,
    "verbose":               args.lgb_verbose,
    "seed":                  args.lgb_random_state,
}
n_train=3411_550
n_val=728789
n_test=728601
M=384
# 1) re-load the memmap arrays
X_train = np.memmap("features/train_emb.npy", dtype="float32", mode="r", shape=(n_train, M))
y_train = np.memmap("features/train_lbl.npy", dtype="int8",   mode="r", shape=(n_train,))

X_val   = np.memmap("features/val_emb.npy",   dtype="float32", mode="r", shape=(n_val,   M))
y_val   = np.memmap("features/val_lbl.npy",   dtype="int8",   mode="r", shape=(n_val,))

# 2) build train_ds (this establishes the bin mapper)
train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

# 3) build val_ds *referencing* train_ds
val_ds = lgb.Dataset(X_val,   label=y_val,   reference=train_ds, free_raw_data=True)

# 4) save them *in order*
train_ds.save_binary("features/train.bin")
val_ds.save_binary(  "features/val.bin")

# now these two .bin files share the exact same bin mapper
bst = lgb.train(
    params,
    train_ds,
    num_boost_round=args.lgb_n_estimators,
    valid_sets=[val_ds],
    callbacks=  [lgb.log_evaluation(args.lgb_verbose), lgb.early_stopping(args.lgb_early_stopping_rounds)]
)