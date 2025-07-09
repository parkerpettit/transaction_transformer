import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union


def split_df_per_card(
    df: pd.DataFrame,
    group_key: str = "Card",
    time_col: str = "trans_datetime",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split each card's history into train/val/test sets.
    """
    train_idx, val_idx, test_idx = [], [], []
    for card, g in df.groupby(by=group_key, sort=False):
        g = g.sort_values(time_col)
        n = len(g)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        train_idx.extend(g.index[:train_end])
        val_idx.extend(g.index[train_end:val_end])
        test_idx.extend(g.index[val_end:])
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def expand_date_features(
    df: pd.DataFrame,
    date_cols: List[str],
    date_parts: Optional[Union[List[str], Dict[str, List[str]]]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand datetime columns into multiple date part features.
    """
    default_parts = ["year", "month", "day", "hour", "minute"]
    new_feats: List[str] = []
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if isinstance(date_parts, dict):
            parts = date_parts.get(col, default_parts)
        else:
            parts = date_parts or default_parts
        for part in parts:
            df[f"{col}_{part}"] = getattr(df[col].dt, part)
            new_feats.append(f"{col}_{part}")
    return df, new_feats


def make_lookup(
    col: pd.Series,
    pad_id: int = 0,
    unk_id: int = 1
) -> Tuple[Dict[str,int], np.ndarray]:
    """
    Build token->id mapping with reserved PAD=0 and UNK=1.
    """
    uniques = sorted(col.dropna().astype(str).unique())
    mapping = {"__PAD__": pad_id, "__UNK__": unk_id}
    mapping.update({tok: i+2 for i, tok in enumerate(uniques)})
    inv = np.empty(len(mapping), dtype=object)
    for tok, idx in mapping.items():
        inv[idx] = tok
    return mapping, inv


def encode_col(
    col: pd.Series,
    mapping: Dict[str,int],
    unk_id: int = 1
) -> np.ndarray:
    """
    Map tokens to IDs; unseen -> UNK.
    """
    return col.astype(str).map(mapping).fillna(unk_id).astype(np.int32).to_numpy()


def preprocess(
    df,
    cat_features: List[str],
    cont_features: List[str],
    date_parts: Optional[Union[List[str], Dict[str,List[str]]]] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Dict[str,Dict], List[str], List[str], StandardScaler
]:
    """
    Load CSV, engineer features, and produce encoded train/val/test splits.
    """
    # 1) Load only needed cols
    # df = pd.read_csv(file)

    # 2) Combine date parts into datetime
    df["trans_datetime"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2) + " " +
        df["Time"],
        errors="coerce"
    )

    # 3) Binary fraud flag
    df["is_fraud"] = df["Is Fraud?"].astype(str).str.lower().map({"yes":1, "no":0})

    # 4) Expand datetime into parts
    df, new_date_feats = expand_date_features(
        df, date_cols=["trans_datetime"], date_parts=date_parts
    )

    # 5) Drop original time cols
    df.drop(columns=["Year","Month","Day","Time","Is Fraud?","trans_datetime"], inplace=True)

    # 6) Split per card
    train_df, val_df, test_df = split_df_per_card(
        df, group_key="Card", time_col="trans_datetime",
        train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    # 7) Update feature lists
    cat_feats = cat_features + new_date_feats
    cont_feats = cont_features.copy()

    # 8) Fit encoders on train only
    encoders: Dict[str, Dict] = {}
    for c in cat_feats:
        mapping, inv = make_lookup(train_df[c])
        encoders[c] = {"map":mapping, "inv":inv}
        train_df[c] = encode_col(train_df[c], mapping)
        val_df[c]   = encode_col(val_df[c], mapping)
        test_df[c]  = encode_col(test_df[c], mapping)

    # 9) Scale cont features
    scaler = StandardScaler().fit(train_df[cont_feats])
    train_df[cont_feats] = scaler.transform(train_df[cont_feats])
    val_df[cont_feats]   = scaler.transform(val_df[cont_feats])
    test_df[cont_feats]  = scaler.transform(test_df[cont_feats])

    return train_df, val_df, test_df, encoders, cat_feats, cont_feats, scaler


def preprocess_for_inference(
    file: str,
    cat_features: List[str],
    cont_features: List[str],
    encoders: Dict[str,Dict],
    scaler: StandardScaler,
    date_parts: Optional[Union[List[str], Dict[str,List[str]]]] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process new data using pre-fitted encoders and scaler.
    """
    df = pd.read_csv(file)
    df["trans_datetime"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2) + " " +
        df["Time"],
        errors="coerce"
    )
    df["is_fraud"] = df["Is Fraud?"].astype(str).str.lower().map({"yes":1, "no":0})
    df, new_date_feats = expand_date_features(
        df, date_cols=["trans_datetime"], date_parts=date_parts
    )
    df.drop(columns=["Year","Month","Day","Time","Is Fraud?","trans_datetime"], inplace=True)
    train_df, val_df, test_df = split_df_per_card(
        df, group_key="Card", time_col="trans_datetime",
        train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    for c in cat_features + new_date_feats:
        mapping = encoders[c]["map"]
        for d in (train_df, val_df, test_df):
            d[c] = encode_col(d[c], mapping)
    for d in (train_df, val_df, test_df):
        d[cont_features] = scaler.transform(d[cont_features])
    return train_df, val_df, test_df
