import os
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from txn_model.data.dataset import TxnDataset, collate_fn
from txn_model.config import ModelConfig, FieldTransformerConfig, SequenceTransformerConfig, LSTMConfig
from txn_model.model import TransactionModel
from txn_model.data.preprocessing import preprocess
from txn_model.logging_utils import configure_logging

logger = configure_logging(__name__)
print("[AR Pretrain] Logger configured")


def slice_batch(batch):
    cat = batch["cat"]
    cont = batch["cont"]
    pad = batch["pad_mask"]
    inp_cat = cat[:, :-1, :]
    inp_cont = cont[:, :-1, :]
    tgt_cat = cat[:, -1, :]
    tgt_cont = cont[:, -1, :]
    inp_mask = pad[:, :-1]
    return inp_cat, inp_cont, inp_mask, tgt_cat, tgt_cont


def build_config(cat_vocab_sizes, cont_features):
    EMB_DIM = 48
    ROW_DIM = 256
    field_cfg = FieldTransformerConfig(d_model=EMB_DIM, n_heads=4, depth=1, ffn_mult=2, dropout=0.10, layer_norm_eps=1e-6, norm_first=True)
    seq_cfg = SequenceTransformerConfig(d_model=ROW_DIM, n_heads=4, depth=4, ffn_mult=2, dropout=0.10, layer_norm_eps=1e-6, norm_first=True)
    lstm_cfg = LSTMConfig(hidden_size=ROW_DIM, num_layers=2, num_classes=2, dropout=0.10)

    return ModelConfig(
        cat_vocab_sizes=cat_vocab_sizes,
        cont_features=cont_features,
        emb_dim=EMB_DIM,
        dropout=0.10,
        padding_idx=0,
        field_transformer=field_cfg,
        sequence_transformer=seq_cfg,
        lstm_config=lstm_cfg,
        total_epochs=0,
    )


def generate_synthetic():
    import pandas as pd
    import numpy as np
    rows = 40
    df = pd.DataFrame({
        "User": np.repeat(np.arange(4), 10),
        "Card": np.random.randint(0, 3, rows),
        "Use Chip": np.random.randint(0, 2, rows),
        "Merchant Name": np.random.randint(0, 4, rows),
        "Merchant City": np.random.randint(0, 3, rows),
        "Merchant State": np.random.randint(0, 3, rows),
        "Zip": np.random.randint(0, 5, rows),
        "MCC": np.random.randint(0, 3, rows),
        "Errors?": np.random.randint(0, 2, rows),
        "Year": np.random.randint(0, 2, rows),
        "Month": np.random.randint(0, 12, rows),
        "Day": np.random.randint(0, 28, rows),
        "Hour": np.random.randint(0, 24, rows),
        "Amount": np.random.randn(rows),
        "is_fraud": np.random.randint(0, 2, rows),
    })
    cat_feats = [
        "User", "Card", "Use Chip", "Merchant Name", "Merchant City",
        "Merchant State", "Zip", "MCC", "Errors?", "Year", "Month",
        "Day", "Hour",
    ]
    cont_feats = ["Amount"]
    encoders = {c: {"inv": [0, 1, 2, 3, 4]} for c in cat_feats}
    return df, encoders, cat_feats, cont_feats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)
    print(f"[AR Pretrain] Using device {device}")

    if args.synthetic:
        train_df, encoders, cat_feats, cont_feats = generate_synthetic()
        print("[AR Pretrain] Using synthetic dataset")
    else:
        if not os.path.exists(args.cache):
            raise FileNotFoundError(args.cache)
        train_df, _, _, encoders, cat_feats, cont_feats, _ = torch.load(args.cache, weights_only=False)
        print(f"[AR Pretrain] Loaded data from {args.cache}")

    cat_sizes = {c: len(encoders[c]["inv"]) for c in cat_feats}
    config = build_config(cat_sizes, cont_feats)
    model = TransactionModel(config).to(device)

    ds = TxnDataset(
        df=train_df,
        group_by=cat_feats[0],
        cat_features=cat_feats,
        cont_features=cont_feats,
        window_size=args.window,
        stride=args.window,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print("[AR Pretrain] DataLoader created")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit_cat = nn.CrossEntropyLoss()
    crit_cont = nn.MSELoss()

    model.train()
    sizes = [cat_sizes[c] for c in cat_feats]
    for ep in range(args.epochs):
        ep_start = time.perf_counter()
        print(f"[AR Pretrain] Epoch {ep + 1}/{args.epochs} start")
        for batch in loader:
            inp_cat, inp_cont, inp_mask, tgt_cat, tgt_cont = slice_batch(batch)
            inp_cat = inp_cat.to(device)
            inp_cont = inp_cont.to(device)
            inp_mask = inp_mask.to(device)
            tgt_cat = tgt_cat.to(device)
            tgt_cont = tgt_cont.to(device)
            logits_cat, pred_cont = model(inp_cat, inp_cont, inp_mask.bool(), mode="ar")
            start = 0
            loss_cat = 0.0
            for i, V in enumerate(sizes):
                end = start + V
                loss_cat += crit_cat(logits_cat[:, start:end], tgt_cat[:, i])
                start = end
            loss = loss_cat + crit_cont(pred_cont, tgt_cont)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "pretrained_backbone.pt")
        logger.info("Epoch %d/%d loss %.4f (%.2fs)", ep + 1, args.epochs, loss.item(), time.perf_counter() - ep_start)
        print(f"[AR Pretrain] Epoch {ep + 1} loss {loss.item():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="processed_data.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    print("[AR Pretrain] Starting main")
    main(args)

