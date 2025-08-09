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
import logging
from logging.handlers import RotatingFileHandler
from transaction_transformer.config.config import ConfigManager
from pandas.util import hash_pandas_object
from transaction_transformer.utils.wandb_utils import init_wandb

logger = logging.getLogger(__name__)


def _setup_logging(job_name: str = "preprocess") -> Path:
    """Configure console INFO and rotating file DEBUG handlers. Returns log file path."""
    log_dir = Path("logs") / job_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{int(time.time())}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # File handler (DEBUG)
    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Avoid duplicate handlers on re-entry
    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)
    return log_path


# NOTE: This script intentionally fits encoders, scalers, and binners
# on legit TRAIN only to avoid any form of data leakage. We build ONE schema
# from legit TRAIN and reuse it for all splits and for both LEGIT and FULL datasets.


def main():
    # ---------------------------------------------------------------------
    #                          LOGGER SETUP
    # ---------------------------------------------------------------------
    log_file = _setup_logging("preprocess")
    logger.info("Starting preprocessing job")

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
    run = init_wandb(cfg, job_type="preprocess", tags=["data"])  # may be None when disabled
    logger.info("Config loaded | wandb_enabled=%s | project=%s", bool(cfg.metrics.wandb), cfg.metrics.wandb_project)

    processed_dir = Path(cfg.model.data.preprocessed_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Artifact-first by default: consume raw dataset from W&B
    if not cfg.model.data.use_local_inputs and cfg.model.data.raw_artifact_name:
        ref = (
            f"{wandb.run.entity}/{wandb.run.project}/{cfg.model.data.raw_artifact_name}:latest"
            if wandb.run else f"{cfg.model.data.raw_artifact_name}:latest"
        )
        logger.info("Loading raw dataset from artifact: %s", ref)
        raw_art = wandb.run.use_artifact(ref) if wandb.run else None
        # Download into the directory specified by config (parent of raw_csv_path)
        raw_root = Path(cfg.model.data.raw_csv_path).parent
        raw_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        raw_dir = Path(raw_art.download(root=str(raw_root))) if raw_art is not None else Path(raw_root)
        logger.info("Raw artifact downloaded to %s in %.2fs", raw_dir, time.time() - t0)
        # Expect the CSV inside the artifact as 'card_transaction.v1.csv'
        raw_csv = raw_dir / Path(cfg.model.data.raw_csv_path).name
        full_train_raw, full_val_raw, full_test_raw = preprocess(raw_csv)
    else:
        logger.info("Running initial CSV preprocessing from local CSV: %s", cfg.model.data.raw_csv_path)
        full_train_raw, full_val_raw, full_test_raw = preprocess(Path(cfg.model.data.raw_csv_path))

    logger.info("Creating legit-only splits ...")
    legit_train_raw: pd.DataFrame = cast(pd.DataFrame, full_train_raw.loc[full_train_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    legit_val_raw: pd.DataFrame = cast(pd.DataFrame, full_val_raw.loc[full_val_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    legit_test_raw: pd.DataFrame = cast(pd.DataFrame, full_test_raw.loc[full_test_raw["is_fraud"] == 0].copy().reset_index(drop=True))
    assert legit_train_raw["is_fraud"].sum() == 0

    logger.info(
        "Split sizes (rows) | full_train=%d full_val=%d full_test=%d | legit_train=%d legit_val=%d legit_test=%d",
        len(full_train_raw), len(full_val_raw), len(full_test_raw), len(legit_train_raw), len(legit_val_raw), len(legit_test_raw)
    )

    # ---------------------------------------------------------------------
    #            BUILD ONE SCHEMA (LEGIT TRAIN) AND REUSE EVERYWHERE
    # ---------------------------------------------------------------------
    logger.info("Fitting encoders and scaler on legit TRAIN only (single schema for all datasets)")
    cat_encoders = get_encoders(legit_train_raw, cat_features)
    scaler = get_scaler(legit_train_raw, cont_features)

    logger.info("Normalizing all splits with scaler fit on legit TRAIN")
    # Normalize LEGIT splits
    legit_train_norm = normalize(legit_train_raw.copy(), scaler, cont_features)
    legit_val_norm = normalize(legit_val_raw.copy(), scaler, cont_features)
    legit_test_norm = normalize(legit_test_raw.copy(), scaler, cont_features)
    # Normalize FULL splits
    full_train_norm = normalize(full_train_raw.copy(), scaler, cont_features)
    full_val_norm = normalize(full_val_raw.copy(), scaler, cont_features)
    full_test_norm = normalize(full_test_raw.copy(), scaler, cont_features)

    logger.info("Fitting per-feature quantile binners on normalized legit TRAIN | num_bins=%d", cfg.model.data.num_bins)
    cont_binners: Dict[str, Any] = {}
    for feat in cont_features:
        t0 = time.time()
        cont_binners[feat] = build_quantile_binner(legit_train_norm[feat])  # type: ignore
        logger.debug("Built binner for %s in %.2fs | bins=%d", feat, time.time() - t0, cont_binners[feat].num_bins)

    schema = FieldSchema(
        cat_features=cat_features,
        cont_features=cont_features,
        cat_encoders=cat_encoders,
        cont_binners=cont_binners,
        time_cat=["Year", "Month", "Day", "Hour"],
        scaler=scaler,
    )

    logger.info("Applying categorical encoders to all splits using single schema")
    # Encode categorical features using the single schema
    legit_train_df = encode_df(legit_train_norm, schema.cat_encoders, schema.cat_features)
    legit_val_df = encode_df(legit_val_norm, schema.cat_encoders, schema.cat_features)
    legit_test_df = encode_df(legit_test_norm, schema.cat_encoders, schema.cat_features)

    full_train_df = encode_df(full_train_norm, schema.cat_encoders, schema.cat_features)
    full_val_df = encode_df(full_val_norm, schema.cat_encoders, schema.cat_features)
    full_test_df = encode_df(full_test_norm, schema.cat_encoders, schema.cat_features)

    # UNK diagnostics per split/feature
    try:
        from transaction_transformer.data.preprocessing.schema import UNK_ID
        def _log_unk(df: pd.DataFrame, name: str):
            for feat in schema.cat_features:
                total = int(len(df))
                unk = int((df[feat] == UNK_ID).sum())
                ratio = (unk / total) if total > 0 else 0.0
                if ratio > 0.05:
                    logger.warning(
                        "UNK rate | split=%s feature=%s | count=%d ratio=%.4f", name, feat, unk, ratio
                    )
                else:
                    logger.debug(
                        "UNK rate | split=%s feature=%s | count=%d ratio=%.4f", name, feat, unk, ratio
                    )
        _log_unk(full_train_df, "full_train")
        _log_unk(full_val_df, "full_val")
        _log_unk(full_test_df, "full_test")
        _log_unk(legit_train_df, "legit_train")
        _log_unk(legit_val_df, "legit_val")
        _log_unk(legit_test_df, "legit_test")
    except Exception as e:
        logger.debug("UNK diagnostics skipped: %s", e)

    # # ---------------------------------------------------------------------
    # #                 ROW-LEVEL LEAKAGE CHECKS (EXACT MATCH)
    # # ---------------------------------------------------------------------
    # logger.info("Leakage check: verifying no exact row overlaps across splits (full and legit families)")

    # def _row_hashes(df: pd.DataFrame) -> pd.Series:
    #     # Ensure consistent column order to hash rows. Convert to string to avoid dtype issues.
    #     cols = list(df.columns)
    #     tmp = df[cols].astype(str)
    #     return hash_pandas_object(tmp, index=False)  # type: ignore[arg-type]

    # # Helper to count overlaps without materializing huge sets for train
    # def _count_overlap(train_df: pd.DataFrame, other_df: pd.DataFrame) -> int:
    #     other_set = set(_row_hashes(other_df).astype("uint64").tolist())
    #     # Stream through train in chunks to control memory
    #     total = 0
    #     chunk = 1_000_000
    #     n = len(train_df)
    #     for start in range(0, n, chunk):
    #         end = min(n, start + chunk)
    #         h = _row_hashes(train_df.iloc[start:end]).astype("uint64")
    #         total += int(h.isin(other_set).sum())
    #     return total

    # # FULL family: train vs val/test and val vs test
    # full_tv = _count_overlap(full_train_df, full_val_df)
    # full_tt = _count_overlap(full_train_df, full_test_df)
    # full_vt = len(set(_row_hashes(full_val_df).astype("uint64").tolist()) & set(_row_hashes(full_test_df).astype("uint64").tolist()))

    # if full_tv or full_tt or full_vt:
    #     msg = (
    #         f"Row overlap detected in FULL splits: train∩val={full_tv}, train∩test={full_tt}, val∩test={full_vt}. "
    #         "This indicates potential data leakage."
    #     )
    #     logger.error(msg)
    #     raise AssertionError(msg)
    # logger.info("Leakage check (FULL) passed: no row overlaps across train/val/test")

    # # LEGIT family: train vs val/test and val vs test
    # legit_tv = _count_overlap(legit_train_df, legit_val_df)
    # legit_tt = _count_overlap(legit_train_df, legit_test_df)
    # legit_vt = len(set(_row_hashes(legit_val_df).astype("uint64").tolist()) & set(_row_hashes(legit_test_df).astype("uint64").tolist()))

    # if legit_tv or legit_tt or legit_vt:
    #     msg = (
    #         f"Row overlap detected in LEGIT splits: train∩val={legit_tv}, train∩test={legit_tt}, val∩test={legit_vt}. "
    #         "This indicates potential data leakage."
    #     )
    #     logger.error(msg)
    #     raise AssertionError(msg)
    # logger.info("Leakage check (LEGIT) passed: no row overlaps across train/val/test")

    # ---------------------------------------------------------------------
    #                           DEBUG: HEAD SNAPSHOTS
    # ---------------------------------------------------------------------
    try:
        import pandas as _pd
        with _pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', None):
            logger.debug("FULL TRAIN head:\n%s", full_train_df.head().to_string(index=False))
            logger.debug("FULL VAL head:\n%s", full_val_df.head().to_string(index=False))
            logger.debug("FULL TEST head:\n%s", full_test_df.head().to_string(index=False))
            logger.debug("LEGIT TRAIN head:\n%s", legit_train_df.head().to_string(index=False))
            logger.debug("LEGIT VAL head:\n%s", legit_val_df.head().to_string(index=False))
            logger.debug("LEGIT TEST head:\n%s", legit_test_df.head().to_string(index=False))
    except Exception:
        logger.debug("Failed to log head() snapshots", exc_info=True)

    # ---------------------------------------------------------------------
    #                                SAVE
    # ---------------------------------------------------------------------
    logger.info("Logging preprocessed datasets and schema to W&B artifact ...")
    run = wandb.run or init_wandb(cfg, job_type="preprocess", tags=["data"])  # idempotent
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
        logger.debug("Writing file into artifact: train.parquet (%d rows)", len(full_train_df))
        full_train_df.to_parquet(f, index=False)
    with processed.new_file("val.parquet", mode="wb") as f:
        logger.debug("Writing file into artifact: val.parquet (%d rows)", len(full_val_df))
        full_val_df.to_parquet(f, index=False)
    with processed.new_file("test.parquet", mode="wb") as f:
        logger.debug("Writing file into artifact: test.parquet (%d rows)", len(full_test_df))
        full_test_df.to_parquet(f, index=False)

    # Also write LEGIT-only splits for pretraining convenience
    with processed.new_file("legit_train.parquet", mode="wb") as f:
        logger.debug("Writing file into artifact: legit_train.parquet (%d rows)", len(legit_train_df))
        legit_train_df.to_parquet(f, index=False)
    with processed.new_file("legit_val.parquet", mode="wb") as f:
        logger.debug("Writing file into artifact: legit_val.parquet (%d rows)", len(legit_val_df))
        legit_val_df.to_parquet(f, index=False)
    with processed.new_file("legit_test.parquet", mode="wb") as f:
        logger.debug("Writing file into artifact: legit_test.parquet (%d rows)", len(legit_test_df))
        legit_test_df.to_parquet(f, index=False)

    # Save schema and helpers
    with processed.new_file("schema.pt", mode="wb") as f:
        logger.debug("Writing file into artifact: schema.pt")
        buf = io.BytesIO()
        torch.save(schema, buf)
        f.write(buf.getvalue())

    # Encoders/binners/scaler saved separately for direct consumption if needed
    with processed.new_file("encoders.pt", mode="wb") as f:
        logger.debug("Writing file into artifact: encoders.pt")
        buf = io.BytesIO()
        torch.save(schema.cat_encoders, buf)
        f.write(buf.getvalue())
    with processed.new_file("binners.pt", mode="wb") as f:
        logger.debug("Writing file into artifact: binners.pt")
        buf = io.BytesIO()
        torch.save(schema.cont_binners, buf)
        f.write(buf.getvalue())
    with processed.new_file("scaler.pt", mode="wb") as f:
        logger.debug("Writing file into artifact: scaler.pt")
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



    # ---------------------------------------------------------------------
    #                         VERIFY DISTRIBUTIONS
    # ---------------------------------------------------------------------
    logger.info("Printing quantile bin distributions for verification (DEBUG) ...")
    import numpy as np

    def print_bin_distributions(df: pd.DataFrame, schema: FieldSchema, dataset_name: str):
        """Prints the bin counts for each continuous feature in a dataframe."""
        logger.debug("--- %s ---", dataset_name)
        for feat in schema.cont_features:
            binner = schema.cont_binners[feat]
            # Data is already normalized, so we can bin it directly
            values = df[feat].to_numpy()
            values = values[~pd.isnull(values)]
            if len(values) == 0:
                logger.debug("Feature '%s': No non-NaN values, skipping.", feat)
                continue
            
            bin_ids = binner.bin(torch.from_numpy(values)).numpy()
            counts = np.bincount(bin_ids, minlength=binner.num_bins)
            logger.debug("Feature '%s' bin counts (total %d bins):", feat, len(counts))
            logger.debug("%s", counts)
        logger.debug("%s", "-" * 40)

    for df, sc, name in [
        (full_train_df, schema, "Full Train"),
        (full_val_df, schema, "Full Val"),
        (full_test_df, schema, "Full Test"),
        (legit_train_df, schema, "Legit Train"),
        (legit_val_df, schema, "Legit Val"),
        (legit_test_df, schema, "Legit Test"),
    ]:
        print_bin_distributions(df, sc, name)

 
    if run is not None:
        try:
            processed.add_file(str(log_file), name="preprocess.log")
            run.log_artifact(processed, aliases=["latest"])
            logger.info("Logged preprocessed-card-v1 artifact with alias [latest]")
        except Exception:
            logger.debug("Failed to upload log file to wandb", exc_info=True)
        wandb.finish()

if __name__ == "__main__":
    main()

