import pathlib
from typing import List, Tuple, cast
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
    # 1a) Strip leading/trailing whitespace from all string/object columns to normalize tokens
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # 2) Generate binary fraud flag
    df["is_fraud"] = df["Is Fraud?"].str.lower().map({"yes": 1, "no": 0}).astype(np.int8)
    df.drop(columns=["Is Fraud?"], inplace=True)
    # GET ONLY LEGIT TRANSACTIONS FOR PRETRAINING
    # df = df[df["is_fraud"] == 0]
    # 3) Parse time, extract hour, drop minutes
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Time"].dt.hour.astype("category")
    # Absolute timestamp in seconds from epoch (normalized later as continuous)
    # NaT will convert to NaN after division
    # ts_ns = df["Time"].astype("int64", copy=False)
    # df["Timestamp"] = (ts_ns / 1e9).astype("float32")
    df.drop(columns=["Time"], inplace=True)

    # 4) Ensure categorical geo fields have appropriate dtypes
    # Zip originates as float64. Convert to pandas nullable integer -> string to avoid FutureWarning.
    zip_num = cast(pd.Series, pd.to_numeric(df["Zip"], errors="coerce")).astype("Int64")
    df["Zip"] = cast(pd.Series, zip_num).astype(pd.StringDtype())
    df["Merchant City"] = df["Merchant City"].astype("string")
    df["Merchant State"] = df["Merchant State"].astype("string")
    df["Errors?"] = df["Errors?"].astype("string")

    # 5) Fill common categorical nulls
    df["Errors?"] = df["Errors?"].fillna("No Error")

    # 6) Normalize online channel: when Merchant City is ONLINE, set missing Zip/State to ONLINE too
    mask_online_city = df["Merchant City"].str.upper().eq("ONLINE")
    df.loc[mask_online_city & df["Zip"].isna(), "Zip"] = "ONLINE"
    df.loc[mask_online_city & df["Merchant State"].isna(), "Merchant State"] = "ONLINE"

    # Convert continuous features
    # For Amount: remove currency symbols/commas even if dtype is pandas StringDtype
    for c in cont_features:
        s = df[c].astype("string")
        s = s.str.replace(r"[\$,]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
        df[c] = df[c].astype("float32")

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