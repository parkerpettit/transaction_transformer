import logging
from typing import List, Dict, Tuple, Any
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path



def get_encoders(df, cat_features):
    # Get encoder from entire dataset  
    encoders: Dict[str, Dict[str, Any]] = {}
    for c in cat_features:
        df[c] = df[c].astype("category")
        cats = df[c].cat.categories
        mapping   = {tok: idx + 2 for idx, tok in enumerate(cats)}   # +2 reserve 0,1
        inv_array = np.array(["__PAD__", "__UNK__", *cats], dtype=object)
        encoders[c] = {"map": mapping, "inv": inv_array}
    return encoders


def encode_df(df, encoders, cat_features):
    # using encoder, map all categorical values to their integer ID
    for c in cat_features:
        mapping = encoders[c]["map"]
        codes = (
            df[c]
            .astype("object")          # <— ensure *not* categorical
            .map(mapping)              # str  → float (with NaN)
            .fillna(1)                 # NaN → __UNK__ code
            .astype(np.int32)          # float → int
        )
        df[c] = codes            # now plain int column
    return df

def get_scaler(df: pd.DataFrame, cont_features: List[str] = ["Amount"]):
    return StandardScaler().fit(df[cont_features].to_numpy())

def normalize(df: pd.DataFrame, scaler: StandardScaler, cont_features: List[str] = ["Amount"]):
    df[cont_features] = scaler.transform(df[cont_features]).astype(np.float32)
    return df


def preprocess(
    file: pathlib.Path,
    cont_features: List[str] = ["Amount"],
    group_key: str = "User",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
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
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Time"].dt.hour.astype("category")
    df.drop(columns=["Time"], inplace=True)

    df["Errors?"] = df["Errors?"].fillna("No Error")
    df["Zip"] = df["Zip"].fillna("Online")

    # strip dollar sign from amounts
    for c in cont_features:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .str.replace(r"[\$,]", "", regex=True)
                .astype("float32")
            )
        else:
            df[c] = pd.to_numeric(df[c], downcast="float")

    # Sort chronologically
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
    return train_df, val_df, test_df

cat_features = [
    "User",
    "Card",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Year",
    "Month",
    "Day",
    "Hour",
]

cont_features = ["Amount"]

print("Preprocessing starting")
full_train_df, full_val_df, full_test_df = preprocess(Path("card_transaction.v1.csv"))
print("Preprocessing done")
print("Getting scaler")

scaler = get_scaler(full_train_df)
# get encoders from entire dataset
print("Getting encoders")

encoders = get_encoders(pd.concat((full_train_df, full_val_df, full_test_df), axis=0), cat_features=cat_features)
    # df = df[df["is_fraud"] == 0]

print("Applying encoders and scalers to dfs")

full_train_df, full_val_df, full_test_df = [
    normalize(encode_df(df.copy(), encoders, cat_features), scaler, cont_features)
    for df in (full_train_df, full_val_df, full_test_df)
]
print("Removing fraud examples from legit dfs")
legit_train, legit_val, legit_test = [
    df[df["is_fraud"] == 0]
    for df in (full_train_df, full_val_df, full_test_df)
]

print("Saving")
import torch
torch.save((full_train_df, full_val_df, full_test_df, encoders, cat_features, cont_features, scaler), "full_processed.pt")
torch.save((legit_train, legit_val, legit_test, encoders, cat_features, cont_features, scaler), "legit_processed.pt")



