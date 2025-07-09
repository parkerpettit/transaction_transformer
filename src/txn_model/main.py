# main.py
import os
import time
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from data.dataset import TxnDataset, collate_fn
from config import ModelConfig, FieldTransformerConfig, SequenceTransformerConfig, LSTMConfig
from data.preprocessing import preprocess
from train import train
import joblib

# Utility logging
def log(msg, start=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start is not None:
        elapsed = time.perf_counter() - start
        print(f"{now} — {msg} (elapsed: {elapsed:.2f}s)")
    else:
        print(f"{now} — {msg}")

# ------------------------------------------------------------------------------
# Load processed data
# ------------------------------------------------------------------------------
log("Starting to load dataframes and misc from joblib")
t0 = time.perf_counter()
cache_path = "/content/drive/MyDrive/summer_urop_25/datasets/processed_data.joblib"
train_df, val_df, test_df, encoders, cat_features, cont_features, scaler = joblib.load(cache_path)
log("Dataframes and misc loaded", t0)

# ------------------------------------------------------------------------------
# Build vocab sizes
# ------------------------------------------------------------------------------
t1 = time.perf_counter()
cat_vocab_sizes = {col: len(enc["inv"]) for col, enc in encoders.items()}
log("Computed cat_vocab_sizes", t1)

# ------------------------------------------------------------------------------
# TxnDataset caching with torch.save / torch.load
# ------------------------------------------------------------------------------
DS_CACHE = "/content/drive/MyDrive/summer_urop_25/datasets/txn_ds_cache.pt"

if os.path.exists(DS_CACHE):
    t_load = time.perf_counter()
    train_ds, val_ds, test_ds = torch.load(DS_CACHE)
    log(f"Loaded full TxnDataset objects via torch.load", t_load)
else:
    log("Building full TxnDataset objects (this runs once)")
    t_build = time.perf_counter()
    train_ds = TxnDataset(
        df=train_df,
        group_by="User",
        cat_features=cat_features,
        cont_features=cont_features,
        window_size=10,
        stride=5
    )
    val_ds = TxnDataset(
        df=val_df,
        group_by="User",
        cat_features=cat_features,
        cont_features=cont_features,
        window_size=10,
        stride=5
    )
    test_ds = TxnDataset(
        df=test_df,
        group_by="User",
        cat_features=cat_features,
        cont_features=cont_features,
        window_size=10,
        stride=5
    )
    log("Built TxnDataset objects", t_build)

    t_save = time.perf_counter()
    torch.save((train_ds, val_ds, test_ds), DS_CACHE)
    log(f"Saved full TxnDataset objects via torch.save to {DS_CACHE}", t_save)

# ------------------------------------------------------------------------------
# Create DataLoaders
# ------------------------------------------------------------------------------
log("Building DataLoaders")
t3 = time.perf_counter()
train_loader = DataLoader(train_ds, shuffle=True, batch_size=32,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                         num_workers=4, collate_fn=collate_fn, pin_memory=True)
log("DataLoaders ready", t3)

# ------------------------------------------------------------------------------
# Define model configuration (unchanged)
# ------------------------------------------------------------------------------
log("Assembling model configuration")
t4 = time.perf_counter()
EMB_DIM = 48
ROW_DIM = 256
HEADS_F = 4
HEADS_S = 4
DEPTH_F = 1
DEPTH_S = 4
FFN_MULT = 2
DROPOUT = 0.10
LN_EPS = 1e-6
LSTM_HID = ROW_DIM
LSTM_NUM_LAYERS = 2
NUM_CLASSES = 1

field_cfg = FieldTransformerConfig(
    d_model=EMB_DIM, n_heads=HEADS_F, depth=DEPTH_F,
    ffn_mult=FFN_MULT, dropout=DROPOUT,
    layer_norm_eps=LN_EPS, norm_first=True
)
sequence_cfg = SequenceTransformerConfig(
    d_model=ROW_DIM, n_heads=HEADS_S, depth=DEPTH_S,
    ffn_mult=FFN_MULT, dropout=DROPOUT,
    layer_norm_eps=LN_EPS, norm_first=True
)
lstm_cfg = LSTMConfig(
    hidden_size=LSTM_HID, num_layers=LSTM_NUM_LAYERS,
    num_classes=NUM_CLASSES, dropout=DROPOUT
)

config = ModelConfig(
    cat_vocab_sizes=cat_vocab_sizes,
    cont_features=cont_features,
    emb_dim=EMB_DIM,
    dropout=DROPOUT,
    padding_idx=0,
    field_transformer=field_cfg,
    sequence_transformer=sequence_cfg,
    lstm_config=lstm_cfg,
    total_epochs=10
)
log("Model configuration assembled", t4)

# ------------------------------------------------------------------------------
# Start training
# ------------------------------------------------------------------------------
log("Starting training")
t5 = time.perf_counter()
train(
    config=config,
    cat_vocab_mapping=cat_vocab_sizes,
    cat_features=cat_features,
    cont_features=cont_features,
    train_loader=train_loader,
    val_loader=val_loader
)
log("Training complete", t5)