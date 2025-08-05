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

# NOTE: This script intentionally fits artefacts (encoders, scalers, binners)
# on *training splits only* to avoid any form of data leakage. We build
# completely separate artefacts for the "full" dataset (fraud + legit) and
# the "legit" dataset (non-fraud only).


def build_schema(train_df: pd.DataFrame, *, cat_features: list[str], cont_features: list[str]):
    """Fit artefacts on ``train_df`` and return a ``FieldSchema`` instance."""
    encoders = get_encoders(train_df, cat_features=cat_features)
    scaler = get_scaler(train_df, cont_features)
    # Currently we only have one continuous feature so we build a single binner.
    # For multiple features you would build one per feature.
    binner = build_quantile_binner(train_df[cont_features[0]])  # type: ignore

    schema = FieldSchema(
        cat_features=cat_features,
        cont_features=cont_features,
        cat_encoders={c: encoders[c] for c in cat_features},
        cont_binners={f: binner for f in cont_features},
        time_cat=["Year", "Month", "Day", "Hour"],
        scaler=scaler,
    )
    return schema


def encode_and_normalize(dfs: list[pd.DataFrame], schema: FieldSchema):
    """Apply categorical encoding + z-score normalisation to each dataframe."""
    return [
        normalize(
            encode_df(df.copy(), schema.cat_encoders, schema.cat_features),
            schema.scaler,
            schema.cont_features,
        )
        for df in dfs
    ]


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    #                          COLUMN DEFINITIONS
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    #                      INITIAL CSV PRE-PROCESSING
    # ---------------------------------------------------------------------
    print("[1] Running initial CSV preprocessing …")
    full_train_raw, full_val_raw, full_test_raw = preprocess(Path("card_transaction.v1.csv"))

    # ---------------------------------------------------------------------
    #                       CREATE LEGIT-ONLY SPLITS
    # ---------------------------------------------------------------------
    legit_train_raw = full_train_raw[full_train_raw["is_fraud"] == 0].copy().reset_index(drop=True)
    legit_val_raw = full_val_raw[full_val_raw["is_fraud"] == 0].copy().reset_index(drop=True)
    legit_test_raw = full_test_raw[full_test_raw["is_fraud"] == 0].copy().reset_index(drop=True)

    # Sanity checks – ensure we actually removed fraud samples
    assert legit_train_raw["is_fraud"].sum() == 0
    assert legit_val_raw["is_fraud"].sum() == 0
    assert legit_test_raw["is_fraud"].sum() == 0

    # ---------------------------------------------------------------------
    #                  FIT + APPLY ARTEFACTS FOR FULL DATASET
    # ---------------------------------------------------------------------
    print("[2] Fitting encoders/scaler for FULL dataset (train split only)…")
    schema_full = build_schema(  # type: ignore[arg-type]
        full_train_raw,
        cat_features=cat_features,
        cont_features=cont_features,
    )

    print("[3] Encoding + normalising FULL dataset …")
    full_train_df, full_val_df, full_test_df = encode_and_normalize(
        [full_train_raw, full_val_raw, full_test_raw], schema_full
    )

    # ---------------------------------------------------------------------
    #                 FIT + APPLY ARTEFACTS FOR LEGIT-ONLY DATASET
    # ---------------------------------------------------------------------
    print("[4] Fitting encoders/scaler for LEGIT dataset (train split only)…")
    schema_legit = build_schema(
        legit_train_raw, # type: ignore
        cat_features=cat_features,
        cont_features=cont_features,
    )

    print("[5] Encoding + normalising LEGIT dataset …")
    legit_train_df, legit_val_df, legit_test_df = encode_and_normalize(
        [legit_train_raw, legit_val_raw, legit_test_raw], schema_legit # type: ignore
    )

    # ---------------------------------------------------------------------
    #                              ASSERTIONS
    # ---------------------------------------------------------------------
    assert len(full_train_df) > len(legit_train_df)
    assert len(full_val_df) > len(legit_val_df)
    assert len(full_test_df) > len(legit_test_df)

    # quick leakage sanity – legit splits should be subsets of full splits
    assert set(legit_train_df.index).issubset(set(full_train_df.index))

    # ---------------------------------------------------------------------
    #                                SAVE
    # ---------------------------------------------------------------------
    print("[6] Saving processed datasets and schemas …")
    import torch

    torch.save(
        (full_train_df, full_val_df, full_test_df, schema_full),
        "full_processed.pt",
    )
    torch.save(
        (legit_train_df, legit_val_df, legit_test_df, schema_legit),
        "legit_processed.pt",
    )

    print("Done")
