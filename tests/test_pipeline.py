import os
import sys
from pathlib import Path
import json
import io
import torch
import pandas as pd
import numpy as np
import pytest


def _write_yaml(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _make_min_schema_and_dfs(n_rows: int = 20):
    # Minimal features for fast tests
    cat_features = ["User", "Year"]
    cont_features = ["Amount"]

    # Build encoders with simple vocab
    from transaction_transformer.data.preprocessing.schema import (
        FieldSchema,
        CatEncoder,
        NumBinner,
    )
    encoders = {
        "User": CatEncoder(mapping={"u0": 3, "u1": 4}, inv=np.array(["[PAD]", "[MASK]", "[UNK]", "u0", "u1"], dtype=object)),
        "Year": CatEncoder(mapping={2020: 3, 2021: 4}, inv=np.array(["[PAD]", "[MASK]", "[UNK]", "2020", "2021"], dtype=object)),
    }
    # Simple 2-bin binner over normalized values
    edges = torch.tensor([-float("inf"), 0.0, float("inf")], dtype=torch.float32)
    binners = {"Amount": NumBinner(edges=edges)}

    # Build small synthetic dataframe
    users = np.random.choice([3, 4], size=n_rows)  # encoded ids
    years = np.random.choice([3, 4], size=n_rows)
    amount = np.random.randn(n_rows).astype(np.float32)
    is_fraud = np.random.choice([0, 1], size=n_rows).astype(np.int64)
    df = pd.DataFrame({
        "User": users,
        "Year": years,
        "Amount": amount,
        "is_fraud": is_fraud,
    })

    # Chronological grouping surrogate
    df = df.reset_index(drop=True)

    # Split into train/val/test simple slices
    n_train = int(0.6 * n_rows)
    n_val = int(0.2 * n_rows)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    # Minimal FieldSchema (no scaler needed for tests)
    from sklearn.preprocessing import StandardScaler
    schema = FieldSchema(
        cat_features=cat_features,
        cont_features=cont_features,
        cat_encoders=encoders,
        cont_binners=binners,
        time_cat=[],
        scaler=StandardScaler(),
    )
    return schema, train_df, val_df, test_df


@pytest.fixture(autouse=True)
def _wandb_offline(monkeypatch):
    # Default tests to offline so they always run; flip to "online" locally if desired
    monkeypatch.setenv("WANDB_MODE", "online")
    yield


def _patch_argv_for_config(tmp_cfg: Path):
    # Ensure config manager picks up our test yaml
    sys.argv = [sys.argv[0], "--config", str(tmp_cfg)]


def test_run_preprocessing_local_csv(tmp_path, monkeypatch):
    # Build a tiny CSV with the columns expected by preprocessing
    csv = tmp_path / "raw.csv"
    data = pd.DataFrame({
        "User": ["u0", "u0", "u1", "u1"],
        "Card": ["c0", "c0", "c1", "c1"],
        "Use Chip": ["Yes", "No", "Yes", "No"],
        "Merchant Name": ["m0", "m0", "m1", "m1"],
        "Merchant City": ["city0", "city0", "city1", "city1"],
        "Merchant State": ["st0", "st0", "st1", "st1"],
        "Zip": ["00000", "00000", "11111", "11111"],
        "MCC": ["mcc0", "mcc0", "mcc1", "mcc1"],
        "Errors?": ["No Error", "No Error", "No Error", "No Error"],
        "Year": [2020, 2020, 2021, 2021],
        "Month": [1, 1, 2, 2],
        "Day": [1, 2, 1, 2],
        "Time": ["12:00", "13:00", "14:00", "15:00"],
        "Hour": [12, 13, 14, 15],
        "Amount": ["$1.00", "$2.00", "$3.00", "$4.00"],
        "Is Fraud?": ["No", "No", "Yes", "No"],
    })
    data.to_csv(csv, index=False)

    # Config using local inputs
    cfg = {
        "model": {
            "mode": "pretrain",
            "field_transformer": {"d_model": 72, "n_heads": 4, "depth": 1},
            "embedding": {"emb_dim": 72, "dropout": 0.0, "freq_encoding_L": 4},
            "data": {
                "use_local_inputs": True,
                "raw_csv_path": str(csv),
                "preprocessed_path": str(tmp_path / "bundle.pt"),
            },
            "training": {
                "device": "cpu",
            },
        },
        "metrics": {
            "wandb": True,
            "wandb_project": "transaction-transformer",
            "wandb_entity": None,
            "run_name": "test-preprocess",
        },
    }
    cfg_path = tmp_path / "pretrain.yaml"
    _write_yaml(cfg_path, cfg)
    _patch_argv_for_config(cfg_path)

    # Execute main; should complete without exceptions (offline W&B)
    from transaction_transformer.data.preprocessing.run_preprocessing import main as preprocess_main
    preprocess_main()


@pytest.mark.parametrize("model_type", ["mlm", "ar"])
def test_pretrain_end_to_end(tmp_path, model_type):
    # Build minimal bundle for pretraining
    schema, train_df, val_df, test_df = _make_min_schema_and_dfs(n_rows=30)
    bundle_path = tmp_path / "preprocessed.pt"
    with io.BytesIO() as buf:
        torch.save((train_df, val_df, test_df, schema), buf)
        bundle_path.write_bytes(buf.getvalue())

    # Pretrain config using local bundle and CPU, small dims for speed
    cfg = {
        "model": {
            "mode": "pretrain",
            "field_transformer": {"d_model": 32, "n_heads": 4, "depth": 1},
            "sequence_transformer": {"d_model": 32, "n_heads": 4, "depth": 1},
            "embedding": {"emb_dim": 32, "dropout": 0.0, "freq_encoding_L": 4},
            "head": {"hidden_dim": 32, "depth": 0, "dropout": 0.0},
            "training": {
                "total_epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "model_type": model_type,
                "device": "cpu",
                "num_workers": 4,
                "max_batches_per_epoch": 1,
            },
            "data": {
                "use_local_inputs": True,
                "preprocessed_path": str(bundle_path),
                "window": 5,
                "stride": 1,
                "group_by": "User",
                "num_bins": 10,
            },
            "pretrain_checkpoint_dir": str(tmp_path / "pretrain_ckpt"),
        },
        "metrics": {
            "wandb": True,
            "wandb_project": "transaction-transformer",
            "wandb_entity": None,
            "run_name": f"test-pretrain-{model_type}",
        },
    }
    cfg_path = tmp_path / "pretrain.yaml"
    _write_yaml(cfg_path, cfg)
    _patch_argv_for_config(cfg_path)

    from transaction_transformer.modeling.training.pretrain import main as pretrain_main
    pretrain_main()

    # Check local checkpoint exports exist
    ckpt_dir = Path(cfg["model"]["pretrain_checkpoint_dir"])
    assert (ckpt_dir / "backbone_last.pt").exists()


def test_finetune_end_to_end_with_pretrained_backbone(tmp_path):
    # 1) Create a pretrained backbone via a tiny pretrain run (AR mode)
    schema, train_df, val_df, test_df = _make_min_schema_and_dfs(n_rows=30)
    bundle_path = tmp_path / "preprocessed_pretrain.pt"
    with io.BytesIO() as buf:
        torch.save((train_df, val_df, test_df, schema), buf)
        bundle_path.write_bytes(buf.getvalue())

    pre_cfg = {
        "model": {
            "mode": "pretrain",
            "field_transformer": {"d_model": 32, "n_heads": 4, "depth": 1},
            "sequence_transformer": {"d_model": 32, "n_heads": 4, "depth": 1},
            "embedding": {"emb_dim": 32, "dropout": 0.0, "freq_encoding_L": 4},
            "head": {"hidden_dim": 32, "depth": 0, "dropout": 0.0},
            "training": {
                "total_epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "model_type": "ar",
                "device": "cpu",
                "num_workers": 4,
                "max_batches_per_epoch": 1,
            },
            "data": {
                "use_local_inputs": True,
                "preprocessed_path": str(bundle_path),
                "window": 5,
                "stride": 1,
                "group_by": "User",
                "num_bins": 10,
            },
            "pretrain_checkpoint_dir": str(tmp_path / "pretrain_ckpt"),
        },
        "metrics": {
            "wandb": True,
            "wandb_project": "transaction-transformer",
            "wandb_entity": None,
            "run_name": "test-pretrain-for-finetune",
        },
    }
    pre_cfg_path = tmp_path / "pretrain.yaml"
    _write_yaml(pre_cfg_path, pre_cfg)
    _patch_argv_for_config(pre_cfg_path)
    from transaction_transformer.modeling.training.pretrain import main as pretrain_main
    pretrain_main()

    # 2) Create finetune bundle (same schema/data ok for smoke test)
    ft_bundle_path = tmp_path / "preprocessed_finetune.pt"
    with io.BytesIO() as buf:
        torch.save((train_df, val_df, test_df, schema), buf)
        ft_bundle_path.write_bytes(buf.getvalue())

    ft_cfg = {
        "model": {
            "mode": "finetune",
            "sequence_transformer": {"d_model": 32, "n_heads": 4, "depth": 1},
            "embedding": {"emb_dim": 32, "dropout": 0.0, "freq_encoding_L": 4},
            "classification": {"hidden_dim": 32, "depth": 1, "dropout": 0.0, "output_dim": 1},
            "training": {
                "total_epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "device": "cpu",
                "num_workers": 4,
                "positive_weight": 1.0,
                "from_scratch": False,
                "max_batches_per_epoch": 1,
            },
            "data": {
                "use_local_inputs": True,
                "preprocessed_path": str(ft_bundle_path),
                "window": 5,
                "stride": 1,
                "group_by": "User",
                "include_all_fraud": True,
            },
            "pretrain_checkpoint_dir": str(tmp_path / "pretrain_ckpt"),
            "finetune_checkpoint_dir": str(tmp_path / "finetune_ckpt"),
        },
        "metrics": {
            "wandb": True,
            "wandb_project": "transaction-transformer",
            "wandb_entity": None,
            "run_name": "test-finetune",
        },
    }
    ft_cfg_path = tmp_path / "finetune.yaml"
    _write_yaml(ft_cfg_path, ft_cfg)
    _patch_argv_for_config(ft_cfg_path)
    from transaction_transformer.modeling.training.finetune import main as finetune_main
    finetune_main()

    # Assert finetune head export exists
    ft_ckpt_dir = Path(ft_cfg["model"]["finetune_checkpoint_dir"])
    assert (ft_ckpt_dir / "clf_head_last.pt").exists()


