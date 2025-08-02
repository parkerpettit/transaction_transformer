import pathlib
from typing import List, Tuple
import pandas as pd
import numpy as np      


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
        d = df[df["split"] == name]
        if not isinstance(d, pd.DataFrame):
            d = pd.DataFrame(d)
        d = d.copy()
        d = d.drop(columns=drop_cols)
        d = d.reset_index(drop=True)
        return d

    train_df = subset("train")
    val_df = subset("val")
    test_df  = subset("test")
    del df  # free RAM
    return train_df, val_df, test_df