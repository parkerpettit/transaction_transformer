import logging
from typing import List, Dict, Tuple, Any
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def preprocess(
    file: pathlib.Path,
    cat_features: List[str],
    cont_features: List[str],
    legit_encoder,
    group_key: str = "User",
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[str, np.ndarray]],
    List[str],
    List[str],
    StandardScaler,
]:
    """Efficient vectorized preprocessing for transaction data."""
    # 1) Read only needed columns
    df = pd.read_csv(file)

    # 2) Generate binary fraud flag
    df["is_fraud"] = df["Is Fraud?"].str.lower().map({"yes": 1, "no": 0}).astype(np.int8)
    df.drop(columns=["Is Fraud?"], inplace=True)
    # GET ONLY LEGIT TRANSACTIONS FOR PRETRAINING
    # df = df[df["is_fraud"] == 0]
    # 3) Parse time, extract hour, drop minutes
    df[time_col] = pd.to_datetime(df[time_col], format="%H:%M", errors="coerce")
    df["Hour"] = df[time_col].dt.hour.astype("category")
    df.drop(columns=[time_col], inplace=True)

    # 4) Downcast continuous features
    for c in cont_features:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .str.replace(r"[\$,]", "", regex=True)
                .astype("float32")
            )
        else:
            df[c] = pd.to_numeric(df[c], downcast="float")

    # 5) Sort chronologically
    df.sort_values(
        by=[group_key, "Year", "Month", "Day", "Hour"],
        ascending=True,
        inplace=True,
        ignore_index=True,
    )

    # 6) Create train/val/test splits ensuring chronological grouping
    df["rank"] = df.groupby(group_key).cumcount()
    df["n_txns"] = df.groupby(group_key)["rank"].transform("max") + 1
    df["split"] = np.where(
        df["rank"] < df["n_txns"] * train_frac, "train",
        np.where(df["rank"] < df["n_txns"] * (train_frac + val_frac), "val", "test")
    )
    # 7) Subset and drop intermediates
    drop_cols = ["rank", "n_txns", "split"]
    def subset(name: str) -> pd.DataFrame:
        d = df[df["split"] == name].copy()
        return d.drop(columns=drop_cols).reset_index(drop=True)

    train_df = subset("train")
    val_df   = subset("val")
    test_df  = subset("test")
    del df  # free RAM

    # 8) Encode categoricals ----------------------------------------------------
    cat_features = [c for c in cat_features if c in train_df.columns]    
    for c in cat_features:
        mapping = legit_encoder[c]["map"]   # <-- pull by column
        for split_df in (train_df, val_df, test_df):
            split_df[c] = (
                split_df[c]
                .astype("object")
                .map(mapping)        # raw → maybe int or NaN
                .fillna(mapping.get("__UNK__", 1))
                .astype(np.int32)
            )

    # 9) Scale continuous and downcast
    cont_features = list(cont_features)
    scaler = StandardScaler().fit(train_df[cont_features].to_numpy())
    for df_ in (train_df, val_df, test_df):
        arr = scaler.transform(df_[cont_features])
        df_[cont_features] = arr.astype(np.float32)

    return train_df, val_df, test_df, legit_encoder, cat_features, cont_features, scaler

import torch
from pathlib import Path

_, _, _, legit_encoders, cat_features, cont_features, _ = torch.load(
    Path("legit_data_only.pt"), weights_only=False # loading processed data from legit only dataset
)

# re process full dataset using legit encoders
train_df, val_df, test_df, encoders, cat_features, cont_features, scaler = preprocess(
    Path("card_transaction.v1.csv"), cat_features=cat_features, cont_features=cont_features, 
    legit_encoder=legit_encoders


)
# save
torch.save((train_df, val_df, test_df, legit_encoders, cat_features, cont_features, scaler), Path("all_data_on_legit_encoders.pt"))