import pandas as pd
from pathlib import Path
from transaction_transformer.data.preprocessing import (
    preprocess,
    FieldSchema,
    get_encoders,
    get_scaler,
    build_quantile_binner,
    normalize,
    encode_df,
)
from typing import Tuple, Dict, Any, cast

# NOTE: This script intentionally fits encoders, scalers, and binners
# on legit TRAIN only to avoid any form of data leakage. We build ONE schema
# from legit TRAIN and reuse it for all splits and for both LEGIT and FULL datasets.


def main():
    # ---------------------------------------------------------------------
    #                          COLUMN DEFINITIONS
    # ---------------------------------------------------------------------
    cat_features = [
        "User", "Card", "Use Chip", "Merchant Name", "Merchant City",
        "Merchant State", "Zip", "MCC", "Errors?", "Year", "Month", "Day", "Hour",
    ]
    cont_features = ["Amount"]

    # ---------------------------------------------------------------------
    #                      INITIAL CSV PRE-PROCESSING
    # ---------------------------------------------------------------------
    print("[1] Running initial CSV preprocessing ...")
    full_train_raw, full_val_raw, full_test_raw = preprocess(Path("data/raw/card_transaction.v1.csv"))

    print("[2] Creating legit-only splits ...")
    legit_train_raw: pd.DataFrame = cast(pd.DataFrame, full_train_raw.loc[full_train_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    legit_val_raw: pd.DataFrame = cast(pd.DataFrame, full_val_raw.loc[full_val_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    legit_test_raw: pd.DataFrame = cast(pd.DataFrame, full_test_raw.loc[full_test_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    assert legit_train_raw["is_fraud"].sum() == 0

    # ---------------------------------------------------------------------
    #            BUILD ONE SCHEMA (LEGIT TRAIN) AND REUSE EVERYWHERE
    # ---------------------------------------------------------------------
    print("\n[SCHEMA] Fitting encoders and scaler on legit TRAIN only (single schema for all datasets)")
    cat_encoders = get_encoders(legit_train_raw, cat_features)
    scaler = get_scaler(legit_train_raw, cont_features)

    print("[SCHEMA] Normalizing all splits with scaler fit on legit TRAIN")
    # Normalize LEGIT splits
    legit_train_norm = normalize(legit_train_raw.copy(), scaler, cont_features)
    legit_val_norm = normalize(legit_val_raw.copy(), scaler, cont_features)
    legit_test_norm = normalize(legit_test_raw.copy(), scaler, cont_features)
    # Normalize FULL splits
    full_train_norm = normalize(full_train_raw.copy(), scaler, cont_features)
    full_val_norm = normalize(full_val_raw.copy(), scaler, cont_features)
    full_test_norm = normalize(full_test_raw.copy(), scaler, cont_features)

    print("[SCHEMA] Fitting per-feature quantile binners on normalized legit TRAIN")
    cont_binners: Dict[str, Any] = {}
    for feat in cont_features:
        cont_binners[feat] = build_quantile_binner(legit_train_norm[feat])  # type: ignore

    schema = FieldSchema(
        cat_features=cat_features,
        cont_features=cont_features,
        cat_encoders=cat_encoders,
        cont_binners=cont_binners,
        time_cat=["Year", "Month", "Day", "Hour"],
        scaler=scaler,
    )

    print("[ENCODE] Applying categorical encoders to all splits using single schema")
    # Encode categorical features using the single schema
    legit_train_df = encode_df(legit_train_norm, schema.cat_encoders, schema.cat_features)
    legit_val_df = encode_df(legit_val_norm, schema.cat_encoders, schema.cat_features)
    legit_test_df = encode_df(legit_test_norm, schema.cat_encoders, schema.cat_features)

    full_train_df = encode_df(full_train_norm, schema.cat_encoders, schema.cat_features)
    full_val_df = encode_df(full_val_norm, schema.cat_encoders, schema.cat_features)
    full_test_df = encode_df(full_test_norm, schema.cat_encoders, schema.cat_features)

    # ---------------------------------------------------------------------
    #                                SAVE
    # ---------------------------------------------------------------------
    print("\n[INFO] Saving processed datasets and single schema ...")
    import torch
    # Ensure output directory exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    torch.save(
        (full_train_df, full_val_df, full_test_df, schema),
        "data/processed/full_processed.pt",
    )
    torch.save(
        (legit_train_df, legit_val_df, legit_test_df, schema),
        "data/processed/legit_processed.pt",
    )
    print("Done.")

    # ---------------------------------------------------------------------
    #                         VERIFY DISTRIBUTIONS
    # ---------------------------------------------------------------------
    print("\n[INFO] Printing quantile bin distributions for verification ...\n")
    import numpy as np

    def print_bin_distributions(df: pd.DataFrame, schema: FieldSchema, dataset_name: str):
        """Prints the bin counts for each continuous feature in a dataframe."""
        print(f"--- {dataset_name} ---")
        for feat in schema.cont_features:
            binner = schema.cont_binners[feat]
            # Data is already normalized, so we can bin it directly
            values = df[feat].to_numpy()
            values = values[~pd.isnull(values)]
            if len(values) == 0:
                print(f"Feature '{feat}': No non-NaN values, skipping.")
                continue
            
            bin_ids = binner.bin(torch.from_numpy(values)).numpy()
            counts = np.bincount(bin_ids, minlength=binner.num_bins)
            print(f"Feature '{feat}' bin counts (total {len(counts)} bins):")
            print(counts)
        print("-" * 40)

    for df, sc, name in [
        (full_train_df, schema, "Full Train"),
        (full_val_df, schema, "Full Val"),
        (full_test_df, schema, "Full Test"),
        (legit_train_df, schema, "Legit Train"),
        (legit_val_df, schema, "Legit Val"),
        (legit_test_df, schema, "Legit Test"),
    ]:
        print_bin_distributions(df, sc, name)

    # # ---------------------------------------------------------------------
    # #                       INSPECT SCHEMA DETAILS
    # # ---------------------------------------------------------------------
    # print("\n[INFO] Inspecting schema details ...")
    # print(f"Number of categorical features: {len(schema.cat_features)}")
    # print(f"Number of continuous features: {len(schema.cont_features)}")
    # print(f"Number of time categorical features: {len(schema.time_cat)}")
    # print(f"Number of bins per continuous feature: {schema.cont_binners['Amount'].num_bins}")   
    # print(f"Categorical fields: {schema.cat_features}")
    # print(f"Continuous fields: {schema.cont_features}")
    # print(f"Time categorical fields: {schema.time_cat}")
    # print(f"Scaler: {schema.scaler}")
    # print(f"Cat encoders: {schema.cat_encoders}")
    # print(f"Cont binners: {schema.cont_binners}")

    # # ---------------------------------------------------------------------
    # #                       INSPECT DATASETS
    # # ---------------------------------------------------------------------
    # print("\n[INFO] Inspecting datasets ...")
    # print(f"Full train shape: {full_train_df.shape}")
    # print(f"Full val shape: {full_val_df.shape}")
    # print(f"Full test shape: {full_test_df.shape}")
    # print(f"Legit train shape: {legit_train_df.shape}")
    # print(f"Legit val shape: {legit_val_df.shape}")
    # print(f"Legit test shape: {legit_test_df.shape}")
    # print(f"Full train columns: {full_train_df.columns.tolist()}")
    # print(f"Full val columns: {full_val_df.columns.tolist()}")
    # print(f"Full test columns: {full_test_df.columns.tolist()}")
    # print(f"Legit train columns: {legit_train_df.columns.tolist()}")
    # print(f"Legit val columns: {legit_val_df.columns.tolist()}")
    # print(f"Legit test columns: {legit_test_df.columns.tolist()}")
    # print(f"Full train head: {full_train_df.head()}")   
    # print(f"Legit train head: {legit_train_df.head()}")
    # print(f"Full train tail: {full_train_df.tail()}")
    # print(f"Legit train tail: {legit_train_df.tail()}")
    # print(f"Full train info: {full_train_df.info()}")
    # print(f"Legit train info: {legit_train_df.info()}")
    # print(f"Full train describe: {full_train_df.describe()}")
    # print(f"Legit train describe: {legit_train_df.describe()}")
    # print(f"Full train isna: {full_train_df.isna().sum()}")
    # print(f"Legit train isna: {legit_train_df.isna().sum()}")
    # print(f"Full train nunique: {full_train_df.nunique()}")
    # print(f"Legit train nunique: {legit_train_df.nunique()}")
    # print(f"Full train value counts: {full_train_df.value_counts()}")
    # print(f"Legit train value counts: {legit_train_df.value_counts()}")
    # print(f"Full train corr: {full_train_df.corr()}")
    # print(f"Legit train corr: {legit_train_df.corr()}")
    # print(f"Full train is_fraud: {full_train_df['is_fraud'].value_counts()}")
    # print(f"Legit train is_fraud: {legit_train_df['is_fraud'].value_counts()}") 
    # print(f"Full val is_fraud: {full_val_df['is_fraud'].value_counts()}")
    # print(f"Legit val is_fraud: {legit_val_df['is_fraud'].value_counts()}")
    # print(f"Full test is_fraud: {full_test_df['is_fraud'].value_counts()}")
    # print(f"Legit test is_fraud: {legit_test_df['is_fraud'].value_counts()}")


if __name__ == "__main__":
    main()

