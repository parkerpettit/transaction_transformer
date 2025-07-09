import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_df_per_user(
    df: pd.DataFrame,
    group_key: str = "cc_num",
    time_col: str = "unix_trans_time",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Chronologically split each user's history into train/val/test sets."""
  train_idx, val_idx, test_idx = [], [], []
  rs = np.random.RandomState(seed)

  for uid, g in df.groupby(by=group_key, sort=False):
    g = g.sort_values(time_col)
    n = len(g)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_idx.extend(g.index[:train_end])
    val_idx.extend(g.index[train_end:val_end])
    test_idx.extend(g.index[val_end:])

  train_df, val_df, test_df = (
    df.loc[train_idx].reset_index(drop=True),
    df.loc[val_idx].reset_index(drop=True),
    df.loc[test_idx].reset_index(drop=True)
  )

  return train_df, val_df, test_df



def expand_date_features(
    df: pd.DataFrame,
    date_cols: list[str],
    date_parts: list[str] | dict[str, list[str]] | None  = None
) -> pd.DataFrame:
    """
    For each column in date_cols, parse it as datetime (if needed)
    and create new columns for each requested calendar field.

    Args:
      df:               your DataFrame
      date_cols:        list of column names to expand
      date_parts:    either
                        - a single list like ["year","month","day","hour"]
                        - a dict mapping col → list of parts
                        - None (defaults to year, month, day, hour)
    Returns:
      df with new columns like "<col>_year", "<col>_month", etc.
    """
    # default parts if nothing specified
    default_parts = ["year", "month", "day", "hour"]
    new_date_feats = []
    for col in date_cols:
        # ensure datetime
        df[col] = pd.to_datetime(df[col], errors="coerce")
        # pick which parts to extract
        if isinstance(date_parts, dict):
            parts = date_parts.get(col, default_parts)
        else:
            parts = date_parts or default_parts
        # expand each part
        for part in parts:
            # getattr(.dt, part) works for year, month, day, hour, minute, second, weekday, etc.
            df[f"{col}_{part}"] = getattr(df[col].dt, part)
            new_date_feats.append(f"{col}_{part}")

    return df, new_date_feats

def preprocess(
    file: str,
    cat_features: list[str],
    cont_features: list[str],
    date_feats_to_expand: list[str] | None = None,
    date_parts: list[str] | dict[str, list[str]] | None  = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict], list[str], list[str], StandardScaler]:
    """Load a CSV, engineer features and produce encoded train/val/test splits."""

    # 1) load & basic clean
    essential = (
        cat_features + cont_features + ["is_fraud"]
        + (date_feats_to_expand or []) + ["dob"]
    )
    df = pd.read_csv(file, usecols=essential)
    df = df[df["is_fraud"] == 0]

    # 2) parse raw datetime columns
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # 3) compute true age in years
    df["age"] = (df["trans_date_trans_time"] - df["dob"]) \
                  .dt.total_seconds() \
                / (365.25 * 24 * 3600)

    # 4) expand each date column into parts
    df, new_date_feats = expand_date_features(df, date_feats_to_expand, date_parts=date_parts)

    # 5) chronological user-level split
    train_df, val_df, test_df = split_df_per_user(
        df,
        group_key="cc_num",
        time_col="trans_date_trans_time"  # this is now a datetime64 column
    )

    drop_cols = (date_feats_to_expand or []) + ["dob"]
    for d in (train_df, val_df, test_df):
        d.drop(columns=drop_cols, inplace=True)

    # new features after feature engineering
    cat_features += new_date_feats  # calendar parts are categoricals
    cont_features += ["age"]

    # 5) fit encoders on train only, reserve PAD=0, UNK=1
    PAD = "<PAD>"
    UNK = "<UNKNOWN>"
    encoders = {}

    # ----------------------- helpers ----------------------------------------
    PAD, UNK = "<PAD>", "<UNKNOWN>"
    PAD_ID,  UNK_ID = 0, 1            # keep the constants in one place


    def make_lookup(col: pd.Series,
                    pad: str = PAD,
                    unk: str = UNK) -> tuple[dict[str, int], np.ndarray]:
        """
        Build a token -> id dict (`mapping`) and an id -> token array (`inv`).

        * pad → 0,  unk → 1
        * the rest are sorted for reproducibility and start at 2
        """
        uniques = sorted(set(col.astype(str)))
        mapping = {pad: PAD_ID, unk: UNK_ID,
                  **{tok: i + 2 for i, tok in enumerate(uniques)}}

        inv = np.empty(len(mapping), dtype=object)
        for tok, idx in mapping.items():
            inv[idx] = tok
        return mapping, inv


    def encode_col(col: pd.Series, mapping: dict[str, int]) -> np.ndarray:
        """Vectorised token→id with UNK fallback."""
        return (
            col.astype(str)
              .map(mapping)          # known → id, unknown → NaN
              .fillna(UNK_ID)        # NaN → UNK id
              .astype(np.int32)
              .to_numpy()
        )


    # ----------------------- main loop --------------------------------------
    encoders: dict[str, dict] = {}

    for c in cat_features:
        mapping, inv = make_lookup(df[c])
        encoders[c] = {"map": mapping, "inv": inv}      # easy to inspect later

        train_df[c] = encode_col(train_df[c], mapping)
        val_df[c]   = encode_col(val_df[c], mapping)
        test_df[c]  = encode_col(test_df[c], mapping)

    # -----------------------------------------------------------------------




    # 6) scale continuous on train, apply to val/test
    scaler = StandardScaler().fit(train_df[cont_features])
    train_df[cont_features] = scaler.transform(train_df[cont_features])
    val_df[cont_features]   = scaler.transform(val_df[cont_features])
    test_df[cont_features]  = scaler.transform(test_df[cont_features])

    return train_df, val_df, test_df, encoders, cat_features, cont_features, scaler






PAD_ID = 0
UNK_ID = 1


def encode_col(col: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    """Map tokens to IDs using provided mapping; unseen tokens → UNK_ID."""
    return (
        col.astype(str)
           .map(mapping)
           .fillna(UNK_ID)
           .astype(np.int32)
           .to_numpy()
    )


def preprocess_for_latents_full(
    file: str,
    cat_features: List[str],
    cont_features: List[str],
    encoders: Dict[str, Dict[str, Any]],  # from genuine-only training
    scaler: StandardScaler,               # fitted on genuine-only training
    date_feats_to_expand: Optional[List[str]] = None,
    date_parts: Optional[Union[List[str], Dict[str, List[str]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Process all transactions (fraud + non-fraud) into chronological train/val/test splits,
    using pre-fitted encoders and scaler from the original genuine-data run.

    Returns:
        train_df, val_df, test_df : DataFrames ready for context-window dataset creation
        cat_features, cont_features: updated feature lists (including date parts and "age")
    """
    # 1) Load only necessary columns
    essential = (
        cat_features + cont_features + ['is_fraud']
        + (date_feats_to_expand or []) + ['dob', 'trans_date_trans_time']
    )
    df = pd.read_csv(file, usecols=essential)
    # 2) Parse datetime columns
    df['trans_date_trans_time'] = pd.to_datetime(
        df['trans_date_trans_time'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # 3) Compute age
    df['age'] = (
        (df['trans_date_trans_time'] - df['dob'])
        .dt.total_seconds() / (365.25 * 24 * 3600)
    )

    # 4) Expand date features
    df, new_date_feats = expand_date_features(
        df, date_feats_to_expand, date_parts=date_parts
    )

    # 5) Chronological user-level split (70/15/15 per card)
    train_df, val_df, test_df = split_df_per_user(
        df,
        group_key='cc_num',
        time_col='trans_date_trans_time',
    )

    # 6) Drop raw date columns
    drop_cols = (date_feats_to_expand or []) + ['dob']
    for d in (train_df, val_df, test_df):
        d.drop(columns=drop_cols, inplace=True)

    # 7) Update feature lists
    cat_features = cat_features + new_date_feats
    cont_features = cont_features + ['age']

    # 8) Encode categoricals in each split using provided mappings
    for col in cat_features:
        mapping = encoders[col]['map']
        for d in (train_df, val_df, test_df):
            d[col] = encode_col(d[col], mapping)

    # 9) Scale continuous features in each split
    for d in (train_df, val_df, test_df):
        d[cont_features] = scaler.transform(d[cont_features])


    return train_df, val_df, test_df, cat_features, cont_features


# Example usage:
# train_df, val_df, test_df, cat_feats, cont_feats = preprocess_for_latents_full(
#     file='transactions.csv',
#     cat_features=orig_cat,
#     cont_features=orig_cont,
#     encoders=saved_encoders,
#     scaler=saved_scaler,
#     date_feats_to_expand=['trans_date_trans_time'],
#     date_parts={'trans_date_trans_time': ['month','day','hour']}
# )

    



