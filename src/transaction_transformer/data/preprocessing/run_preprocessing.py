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
import wandb
import json
import io
import torch
import time
from transaction_transformer.config.config import ConfigManager

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
    # Load config to determine artifact behavior
    cfg = ConfigManager(config_path="pretrain.yaml").config
    if cfg.metrics.wandb and wandb.run is None:
        wandb.init(project=cfg.metrics.wandb_project, name="preprocess", config=cfg.to_dict(), job_type="preprocess")

    processed_dir = Path(cfg.model.data.preprocessed_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Artifact-first by default: consume raw dataset from W&B
    if not cfg.model.data.use_local_inputs and cfg.model.data.raw_artifact_name:
        ref = f"{wandb.run.entity}/{wandb.run.project}/{cfg.model.data.raw_artifact_name}:latest" if wandb.run else f"{cfg.model.data.raw_artifact_name}:latest"
        print(f"[1] Loading raw dataset from artifact: {ref}")
        raw_art = wandb.run.use_artifact(ref)  # type: ignore[arg-type]
        # Download into the directory specified by config (parent of raw_csv_path)
        raw_root = Path(cfg.model.data.raw_csv_path).parent
        raw_root.mkdir(parents=True, exist_ok=True)
        raw_dir = Path(raw_art.download(root=str(raw_root)))
        # Expect the CSV inside the artifact as 'card_transaction.v1.csv'
        raw_csv = raw_dir / Path(cfg.model.data.raw_csv_path).name
        full_train_raw, full_val_raw, full_test_raw = preprocess(raw_csv)
    else:
        print("[1] Running initial CSV preprocessing from local CSV ...")
        full_train_raw, full_val_raw, full_test_raw = preprocess(Path(cfg.model.data.raw_csv_path))

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
    print("\n[INFO] Logging preprocessed datasets and schema to W&B artifact ...")
    run = wandb.run or wandb.init(project=cfg.metrics.wandb_project, name="preprocess", config=cfg.to_dict(), job_type="preprocess")
    processed = wandb.Artifact(
        name="preprocessed-card-v1",
        type="dataset",
        description="Preprocessed splits, schema, encoders, binners, scaler for card v1",
        metadata={
            "job_type": "preprocess",
            "created_at": time.time(),
        },
    )
    # Write FULL dataframes as parquet directly into the artifact (binary mode)
    with processed.new_file("train.parquet", mode="wb") as f:
        full_train_df.to_parquet(f, index=False)
    with processed.new_file("val.parquet", mode="wb") as f:
        full_val_df.to_parquet(f, index=False)
    with processed.new_file("test.parquet", mode="wb") as f:
        full_test_df.to_parquet(f, index=False)

    # Also write LEGIT-only splits for pretraining convenience
    with processed.new_file("legit_train.parquet", mode="wb") as f:
        legit_train_df.to_parquet(f, index=False)
    with processed.new_file("legit_val.parquet", mode="wb") as f:
        legit_val_df.to_parquet(f, index=False)
    with processed.new_file("legit_test.parquet", mode="wb") as f:
        legit_test_df.to_parquet(f, index=False)

    # Save schema and helpers
    with processed.new_file("schema.pt", mode="wb") as f:
        buf = io.BytesIO()
        torch.save(schema, buf)
        f.write(buf.getvalue())

    # Encoders/binners/scaler saved separately for direct consumption if needed
    with processed.new_file("encoders.pt", mode="wb") as f:
        buf = io.BytesIO()
        torch.save(schema.cat_encoders, buf)
        f.write(buf.getvalue())
    with processed.new_file("binners.pt", mode="wb") as f:
        buf = io.BytesIO()
        torch.save(schema.cont_binners, buf)
        f.write(buf.getvalue())
    with processed.new_file("scaler.pt", mode="wb") as f:
        buf = io.BytesIO()
        torch.save(schema.scaler, buf)
        f.write(buf.getvalue())

    # Add preprocessing metadata
    preprocess_meta = {
        "seed": cfg.metrics.seed,
        "window": cfg.model.data.window,
        "stride": cfg.model.data.stride,
        "group_by": cfg.model.data.group_by,
        "cat_features": schema.cat_features,
        "cont_features": schema.cont_features,
        "time_cat": getattr(schema, "time_cat", []),
        "special_ids": {
            "pad": cfg.model.data.padding_idx,
            "mask": cfg.model.data.mask_idx,
            "unk": cfg.model.data.unk_idx,
        },
    }
    with processed.new_file("preprocess_meta.json", mode="w") as f:
        f.write(json.dumps(preprocess_meta, indent=2))

    # Also copy files to data/processed for local overrides
    import shutil
    for src_name in [
        "train.parquet", "val.parquet", "test.parquet",
        "legit_train.parquet", "legit_val.parquet", "legit_test.parquet",
        "schema.pt", "encoders.pt", "binners.pt", "scaler.pt", "preprocess_meta.json",
    ]:
        src_path = Path(run.dir) / "artifacts"  # not reliable; copy from temp created via new_file is non-trivial
    # We rely on artifact download in subsequent steps to populate local cache when needed.

    run.log_artifact(processed, aliases=["latest", f"run-{run.id}"])
    print("Logged preprocessed-card-v1 artifact with aliases [latest, run-<id>]")
    # Finish run explicitly to flush uploads and close the run
    wandb.finish()

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

