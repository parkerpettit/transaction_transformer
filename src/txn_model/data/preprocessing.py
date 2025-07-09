import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

def preprocess(
    file: str,
    cat_features: List[str],
    cont_features: List[str],
    group_key: str = "User",
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Dict[str, Dict[str, np.ndarray]],
    List[str], List[str], StandardScaler
]:
    """
    Fast, vectorized preprocessing:
      - binary label from "Is Fraud?"
      - parse `time_col` into Hour
      - vectorized split by `group_key` into train/val/test
      - pandas categorical codes (+2 for PAD/UNK) for all cat_features + Hour
      - one-shot StandardScaler on all cont_features
    """
    # 1) load
    df = pd.read_feather(file)

    # 2) binary fraud flag
    df["is_fraud"] = df["Is Fraud?"].str.lower().map({"yes":1, "no":0})

    # 3) extract Hour and drop original time & fraud columns
    df[time_col] = pd.to_datetime(df[time_col], format="%H:%M", errors="coerce")
    df["Hour"]  = df[time_col].dt.hour.astype("category")
    df.drop(columns=[time_col, "Is Fraud?"], inplace=True)

    # 4) clean any stringy cont_features → numeric
    for c in cont_features:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(
                df[c].astype(str)
                     .str.replace(r"[\$,]", "", regex=True),
                errors="coerce"
            )

    # 5) vectorized train/val/test split by group_key
    df["rank"]   = df.groupby(group_key).cumcount()
    df["n_txns"] = df.groupby(group_key)["rank"].transform("max") + 1

    df["split"] = np.where(
        df["rank"] <  df["n_txns"] * train_frac, "train",
    np.where(
        df["rank"] <  df["n_txns"] * (train_frac + val_frac), "val",
        "test"
    ))

    # 6) assemble final feature lists
    cat_feats  = list(cat_features) + ["Hour"]
    cont_feats = list(cont_features)

    # 7) encode categoricals via pandas .cat.codes (+2), build map & inv arrays
    encoders: Dict[str, Dict[str, np.ndarray]] = {}
    for c in cat_feats:
        df[c] = df[c].astype("category")
        codes = df[c].cat.codes.to_numpy(dtype=np.int32) + 2
        inv   = df[c].cat.categories.to_numpy()
        mapping = {tok: idx+2 for idx, tok in enumerate(inv)}
        inv_array = np.empty(len(mapping) + 2, dtype=object)
        inv_array[0], inv_array[1] = "__PAD__", "__UNK__"
        inv_array[2:] = inv
        df[c] = codes
        encoders[c] = {"map": mapping, "inv": inv_array}

    # 8) fit & apply StandardScaler to all cont_feats
    scaler = StandardScaler().fit(df[cont_feats].to_numpy())
    df[cont_feats] = scaler.transform(df[cont_feats].to_numpy())

    # 9) split out DataFrames and drop helper cols
    drop_cols = ["rank", "n_txns", "split"]
    def subset(name: str) -> pd.DataFrame:
        subset_df = df[df["split"] == name].copy()
        return subset_df.drop(columns=drop_cols).reset_index(drop=True)

    train_df = subset("train")
    val_df   = subset("val")
    test_df  = subset("test")

    return train_df, val_df, test_df, encoders, cat_feats, cont_feats, scaler
