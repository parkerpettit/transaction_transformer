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
from typing import Tuple

# NOTE: This script intentionally fits encoders, scalers, and binners
# on *training splits only* to avoid any form of data leakage. We build
# completely separate artifacts for the "full" dataset (fraud + legit) and
# the "legit" dataset (non-fraud only).


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
    legit_train_raw = full_train_raw[full_train_raw["is_fraud"] == 0].copy().reset_index(drop=True)
    legit_val_raw = full_val_raw[full_val_raw["is_fraud"] == 0].copy().reset_index(drop=True)
    legit_test_raw = full_test_raw[full_test_raw["is_fraud"] == 0].copy().reset_index(drop=True)
    assert legit_train_raw["is_fraud"].sum() == 0

    # ---------------------------------------------------------------------
    #       PROCESS FULL & LEGIT DATASETS WITH CORRECTED PIPELINE
    # ---------------------------------------------------------------------

    def process_dataset(
        train_raw: pd.DataFrame, 
        val_raw: pd.DataFrame, 
        test_raw: pd.DataFrame, 
        name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FieldSchema]:
        
        print(f"\n[{name.upper()}] Processing dataset ...")

        # Step 1: Fit encoders and scaler on RAW training data
        print(f"  - Fitting encoders and scaler on raw {name} training data")
        cat_encoders = get_encoders(train_raw, cat_features)
        scaler = get_scaler(train_raw, cont_features)

        # Step 2: Normalize all data splits
        print(f"  - Normalizing all {name} splits")
        train_norm = normalize(train_raw.copy(), scaler, cont_features)
        val_norm = normalize(val_raw.copy(), scaler, cont_features)
        test_norm = normalize(test_raw.copy(), scaler, cont_features)

        # Step 3: Fit binner on NORMALIZED training data
        print(f"  - Fitting quantile binner on normalized {name} training data")
        binner = build_quantile_binner(train_norm[cont_features[0]])  # type: ignore

        # Step 4: Create final schema with all fitted artifacts
        schema = FieldSchema(
            cat_features=cat_features,
            cont_features=cont_features,
            cat_encoders=cat_encoders,
            cont_binners={f: binner for f in cont_features},
            time_cat=["Year", "Month", "Day", "Hour"],
            scaler=scaler,
        )

        # Step 5: Apply categorical encoding to all (now normalized) splits
        print(f"  - Applying categorical encoding to all {name} splits")
        train_df = encode_df(train_norm, schema.cat_encoders, schema.cat_features)
        val_df = encode_df(val_norm, schema.cat_encoders, schema.cat_features)
        test_df = encode_df(test_norm, schema.cat_encoders, schema.cat_features)
        
        return train_df, val_df, test_df, schema

    # Process both full and legit datasets
    full_train_df, full_val_df, full_test_df, schema_full = process_dataset(
        full_train_raw, full_val_raw, full_test_raw, "full"
    )
    legit_train_df, legit_val_df, legit_test_df, schema_legit = process_dataset(
        legit_train_raw, legit_val_raw, legit_test_raw, "legit"  # type: ignore
    )

    # ---------------------------------------------------------------------
    #                                SAVE
    # ---------------------------------------------------------------------
    print("\n[INFO] Saving processed datasets and schemas ...")
    import torch
    torch.save(
        (full_train_df, full_val_df, full_test_df, schema_full),
        "data/processed/full_processed.pt",
    )
    torch.save(
        (legit_train_df, legit_val_df, legit_test_df, schema_legit),
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

    for df, schema, name in [
        (full_train_df, schema_full, "Full Train"),
        (full_val_df, schema_full, "Full Val"),
        (legit_train_df, schema_legit, "Legit Train"),
    ]:
        print_bin_distributions(df, schema, name)



if __name__ == "__main__":
    main()

