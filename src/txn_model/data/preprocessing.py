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
    df.sort_values(by=["Year", "Month", "Day", "Hour"], ascending=True, inplace=True, ignore_index=True)

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

    # 8) Encode categoricals
    cat_features = list(cat_features) + ["Hour"]
    encoders: Dict[str, Dict[str, Any]] = {}
    for c in cat_features:
        train_df[c] = train_df[c].astype("category")
        cats = train_df[c].cat.categories
        mapping = {tok: idx + 2 for idx, tok in enumerate(cats)}
        inv_array = np.array(["__PAD__", "__UNK__"] + list(cats), dtype=object)
        encoders[c] = {"map": mapping, "inv": inv_array}
        for split_df in (train_df, val_df, test_df):
            split_df[c] = split_df[c].map(mapping).fillna(1).astype(np.int32)

    # 9) Scale continuous and downcast
    cont_features = list(cont_features)
    scaler = StandardScaler().fit(train_df[cont_features].to_numpy())
    for df_ in (train_df, val_df, test_df):
        arr = scaler.transform(df_[cont_features])
        df_[cont_features] = arr.astype(np.float32)

    return train_df, val_df, test_df, encoders, cat_features, cont_features, scaler

