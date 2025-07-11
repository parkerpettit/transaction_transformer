import os
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader

from data.dataset import TxnDataset, collate_fn
from config import ModelConfig, FieldTransformerConfig, SequenceTransformerConfig, LSTMConfig
from data.preprocessing import preprocess
from train import train
from logging_utils import configure_logging

logger = configure_logging(__name__)


PD_CACHE = "/content/drive/MyDrive/summer_urop_25/datasets/processed_data.pt"
DATASET_PATH = "/content/drive/MyDrive/summer_urop_25/datasets/tabformer.feather"

cat_features = [
    "User", "Card", "Use Chip", "Merchant Name", "Merchant City",
    "Merchant State", "Zip", "MCC", "Errors?", "Year", "Month", "Day", "Hour",
]
cont_features = ["Amount"]


def main() -> None:
    if os.path.exists(PD_CACHE):
        t0 = time.perf_counter()
        train_df, val_df, test_df, encoders, cat_features_loaded, cont_features_loaded, scaler = torch.load(PD_CACHE, weights_only=False)
        logger.info("Loaded processed data (%.2fs)", time.perf_counter() - t0)
        print(f"[Main] Loaded cached processed data in {time.perf_counter() - t0:.2f}s")
    else:
        logger.info("Running full preprocess pipeline")
        print("[Main] Running full preprocess pipeline")
        t0 = time.perf_counter()
        train_df, val_df, test_df, encoders, cat_features_loaded, cont_features_loaded, scaler = preprocess(
            DATASET_PATH, cat_features, cont_features
        )
        logger.info("Preprocess completed in %.2fs", time.perf_counter() - t0)
        print(f"[Main] Preprocess completed in {time.perf_counter() - t0:.2f}s")
        torch.save((train_df, val_df, test_df, encoders, cat_features_loaded, cont_features_loaded, scaler), PD_CACHE)
        logger.info("Saved processed data to %s", PD_CACHE)
        print(f"[Main] Saved processed data to {PD_CACHE}")

    cat_vocab_sizes = {col: len(enc["inv"]) for col, enc in encoders.items()}

    logger.info("Building TxnDataset objects")
    print("[Main] Building TxnDataset objects")
    t_ds = time.perf_counter()
    train_ds = TxnDataset(train_df, group_by="User", cat_features=cat_features_loaded, cont_features=cont_features_loaded, window_size=100, stride=50)
    val_ds = TxnDataset(val_df, group_by="User", cat_features=cat_features_loaded, cont_features=cont_features_loaded, window_size=100, stride=50)
    test_ds = TxnDataset(test_df, group_by="User", cat_features=cat_features_loaded, cont_features=cont_features_loaded, window_size=100, stride=50)
    logger.info("Datasets ready in %.2fs", time.perf_counter() - t_ds)
    print(f"[Main] Datasets ready in {time.perf_counter() - t_ds:.2f}s")

    logger.info("Creating DataLoaders")
    print("[Main] Creating DataLoaders")
    t_dl = time.perf_counter()
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=650, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=650, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=650, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    logger.info("DataLoaders ready in %.2fs", time.perf_counter() - t_dl)
    print(f"[Main] DataLoaders ready in {time.perf_counter() - t_dl:.2f}s")

    logger.info("Assembling model configuration")
    print("[Main] Assembling model configuration")
    EMB_DIM = 48
    ROW_DIM = 256
    field_cfg = FieldTransformerConfig(d_model=EMB_DIM, n_heads=4, depth=1, ffn_mult=2, dropout=0.10, layer_norm_eps=1e-6, norm_first=True)
    sequence_cfg = SequenceTransformerConfig(d_model=ROW_DIM, n_heads=4, depth=4, ffn_mult=2, dropout=0.10, layer_norm_eps=1e-6, norm_first=True)
    lstm_cfg = LSTMConfig(hidden_size=ROW_DIM, num_layers=2, num_classes=2, dropout=0.10)

    config = ModelConfig(
        cat_vocab_sizes=cat_vocab_sizes,
        cont_features=cont_features_loaded,
        emb_dim=EMB_DIM,
        dropout=0.10,
        padding_idx=0,
        field_transformer=field_cfg,
        sequence_transformer=sequence_cfg,
        lstm_config=lstm_cfg,
        total_epochs=10,
    )

    logger.info("Starting training")
    print("[Main] Starting training")
    t_train = time.perf_counter()
    train(
        config=config,
        cat_vocab_mapping=cat_vocab_sizes,
        cat_features=cat_features_loaded,
        cont_features=cont_features_loaded,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    logger.info("Training complete in %.2fs", time.perf_counter() - t_train)
    print(f"[Main] Training complete in {time.perf_counter() - t_train:.2f}s")


if __name__ == "__main__":
    main()

