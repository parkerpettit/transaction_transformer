import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess(
    file: str,
    cat_features: List[str],
    cont_features: List[str],
    group_key: str = "User",
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]], List[str], List[str], StandardScaler]:
    """Vectorized preprocessing for the transaction dataset."""
    logger.info("Loading raw data from %s", file)
    df = pd.read_feather(file)
    logger.debug("Raw data shape: %s", df.shape)

    logger.info("Generating binary fraud flag")
    df["is_fraud"] = df["Is Fraud?"].str.lower().map({"yes": 1, "no": 0})

    logger.info("Parsing time column %s", time_col)
    df[time_col] = pd.to_datetime(df[time_col], format="%H:%M", errors="coerce")
    df["Hour"] = df[time_col].dt.hour.astype("category")
    df.drop(columns=[time_col, "Is Fraud?"], inplace=True)

    logger.info("Converting continuous features")
    for c in cont_features:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")

    logger.info("Splitting dataset by %s", group_key)
    df["rank"] = df.groupby(group_key).cumcount()
    df["n_txns"] = df.groupby(group_key)["rank"].transform("max") + 1
    df["split"] = np.where(
        df["rank"] < df["n_txns"] * train_frac,
        "train",
        np.where(df["rank"] < df["n_txns"] * (train_frac + val_frac), "val", "test"),
    )

    logger.info("Creating train/val/test subsets")
    cat_feats = list(cat_features) + ["Hour"]
    cont_feats = list(cont_features)
    drop_cols = ["rank", "n_txns", "split"]

    def subset(name: str) -> pd.DataFrame:
        subset_df = df[df["split"] == name].copy()
        return subset_df.drop(columns=drop_cols).reset_index(drop=True)

    train_df = subset("train")
    val_df = subset("val")
    test_df = subset("test")

    logger.info("Encoding categorical features")
    encoders: Dict[str, Dict[str, np.ndarray]] = {}
    for c in cat_feats:
        train_df[c] = train_df[c].astype("category")
        cats = train_df[c].cat.categories
        mapping = {tok: idx + 2 for idx, tok in enumerate(cats)}
        inv_array = np.array(["__PAD__", "__UNK__"] + list(cats), dtype=object)
        encoders[c] = {"map": mapping, "inv": inv_array}
        for split_df in (train_df, val_df, test_df):
            split_df[c] = split_df[c].astype("object")
            codes = split_df[c].map(mapping).fillna(1).astype(int)
            split_df[c] = codes

    logger.info("Fitting StandardScaler on continuous features")
    scaler = StandardScaler().fit(train_df[cont_feats].to_numpy())
    train_df[cont_feats] = scaler.transform(train_df[cont_feats])
    val_df[cont_feats] = scaler.transform(val_df[cont_feats])
    test_df[cont_feats] = scaler.transform(test_df[cont_feats])

    logger.info(
        "Finished preprocessing -> train %d rows, val %d rows, test %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df, encoders, cat_feats, cont_feats, scaler

