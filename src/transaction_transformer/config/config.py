"""
Base configuration classes for the feature prediction transformer.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import argparse


@dataclass
class TransformerConfig:
    """Configuration for transformer components (field or sequence)."""

    d_model: int = 512
    n_heads: int = 8
    depth: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    norm_first: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layer."""

    emb_dim: int = 72  # Should match field transformer d_model
    dropout: float = 0.1
    padding_idx: int = 0
    freq_encoding_L: int = 8  # For continuous features


@dataclass
class ClassificationConfig:
    """Configuration for classification head."""

    hidden_dim: int = 512
    depth: int = 0  # Number of layers in the MLP. 0 is a linear layer
    dropout: float = 0.1
    output_dim: int = 1


@dataclass
class HeadConfig:
    """Configuration for prediction heads."""

    hidden_dim: int = 512
    depth: int = 0  # Number of layers in the MLP. 0 is a linear layer
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Basic training
    total_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1.0e-4

    # Training model_type
    model_type: str = "mlm"  # "mlm" or "ar"

    # MLM-specific parameters
    p_field: float = 0.15  # Field masking probability
    p_row: float = 0.10  # Row masking probability

    # Mixed precision
    use_amp: bool = False


    # Checkpointing and logging
    early_stopping_patience: int = 5

    # Device and workers
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 0

    # Class imbalance handling
    positive_weight: float = 1.0  # Weight for positive class in binary classification

    # Optional debug cap for faster iteration
    max_batches_per_epoch: Optional[int] = None

    # Checkpoint/resume behavior
    resume: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data paths
    preprocessed_path: str = "data/processed/legit_processed.pt"
    raw_csv_path: str = (
        "data/raw/card_transaction.v1.csv"  # local fallback/raw upload source
    )
    # Artifact-first behavior (download ':latest' each run unless overridden)
    use_local_inputs: bool = False
    raw_artifact_name: Optional[str] = "raw-card-transactions"
    preprocessed_artifact_name: Optional[str] = "preprocessed-card"

    # Sequence parameters
    window: int = 10
    stride: int = 5

    # Binning configuration
    num_bins: int = 100  # For continuous features

    # Dataset parameters
    group_by: str = "User"  # Column to group sequences by
    include_all_fraud: bool = False

    # Special tokens (should match schema)
    padding_idx: int = 0
    mask_idx: int = 1
    unk_idx: int = 2
    ignore_idx: int = -100


@dataclass
class ModelConfig:
    """Main model configuration."""

    # Model architecture
    mode: str = "pretrain"  # "pretrain", "finetune"
    head_type: str = "mlp"  # "mlp", "lstm"
    # Transformer configurations
    field_transformer: TransformerConfig = field(default_factory=TransformerConfig)

    sequence_transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # Classification configuration
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    # Head configuration
    head: HeadConfig = field(default_factory=HeadConfig)

    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # Row projection
    row_types: int = 1  # Number of different row types

    # Model parameters
    freeze_embedding: bool = False

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)

    # Paths
    pretrain_checkpoint_dir: str = "data/models/pretrained"
    finetune_checkpoint_dir: str = "data/models/finetuned"


@dataclass
class MetricsConfig:
    """Configuration for experiment tracking and logging."""

    run_name: str = "pretrain"
    run_id: Optional[str] = None

    # Logging
    wandb_project: str = "feature-predictor"

    # Reproducibility (stored in metadata)
    seed: int = 42


@dataclass
class Config:
    """Complete configuration for the feature prediction transformer."""

    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate transformer configurations
        if self.model.field_transformer.d_model != self.model.embedding.emb_dim:
            raise ValueError(
                f"Field transformer d_model ({self.model.field_transformer.d_model}) "
                f"must match embedding emb_dim ({self.model.embedding.emb_dim})"
            )

        # Validate training model_type
        if self.model.training.model_type not in ["mlm", "ar"]:
            raise ValueError(
                f"Training model_type must be 'mlm' or 'ar', got {self.model.training.model_type}"
            )

        # Validate masking probabilities
        if not 0 <= self.model.training.p_field <= 1:
            raise ValueError(
                f"p_field must be between 0 and 1, got {self.model.training.p_field}"
            )
        if not 0 <= self.model.training.p_row <= 1:
            raise ValueError(
                f"p_row must be between 0 and 1, got {self.model.training.p_row}"
            )

        # Validate device
        if self.model.training.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(
                f"Device must be 'auto', 'cpu', or 'cuda', got {self.model.training.device}"
            )

    def get_device(self) -> str:
        """Get the device to use for training."""
        if self.model.training.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model.training.device

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from (possibly nested) dictionary by overriding the default dataclass values.
        This performs a recursive update so missing keys keep their default values, while specified
        keys in the YAML/CLI dict overwrite the defaults.
        """
        import dataclasses

        def _update_dataclass(dc_obj: Any, updates: Dict[str, Any]):
            """Recursively update a dataclass instance from a dict of updates."""
            for key, value in updates.items():
                if not hasattr(dc_obj, key):
                    continue  # ignore unknown keys to stay robust
                current_val = getattr(dc_obj, key)
                if dataclasses.is_dataclass(current_val) and isinstance(value, dict):
                    _update_dataclass(current_val, value)
                else:
                    setattr(dc_obj, key, value)

        # 1) Start from defaults
        cfg = cls()
        # 2) Recursively apply overrides
        _update_dataclass(cfg, config_dict)
        # 3) Validate and return
        cfg._validate_config()
        return cfg


class ConfigManager:
    """Manages configuration loading, merging, and validation."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config(self.config_path)

    def load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def parse_cli_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Transaction Transformer Training")

        # Configuration file
        parser.add_argument("--config", type=str, help="Path to configuration file")

        # Model parameters
        parser.add_argument(
            "--model-type",
            type=str,
            choices=["mlm", "ar"],
            help="Pretraining model_type (MLM or autoregressive)",
        )

        # Training parameters
        parser.add_argument("--batch-size", type=int, help="Batch size")
        parser.add_argument("--learning-rate", type=float, help="Learning rate")
        parser.add_argument("--total-epochs", type=int, help="Total training epochs")
        parser.add_argument(
            "--device", type=str, help="Device to use (cpu, cuda, auto)"
        )

        # Transformer parameters
        parser.add_argument(
            "--field-d-model", type=int, help="Field transformer d_model"
        )
        parser.add_argument(
            "--field-n-heads", type=int, help="Field transformer number of heads"
        )
        parser.add_argument("--field-depth", type=int, help="Field transformer depth")
        parser.add_argument(
            "--seq-d-model", type=int, help="Sequence transformer d_model"
        )
        parser.add_argument(
            "--seq-n-heads", type=int, help="Sequence transformer number of heads"
        )
        parser.add_argument("--seq-depth", type=int, help="Sequence transformer depth")

        # Data parameters
        parser.add_argument("--window", type=int, help="Sequence window size")
        parser.add_argument("--stride", type=int, help="Stride between windows")
        parser.add_argument(
            "--num-bins", type=int, help="Number of bins for continuous features"
        )

        # MLM-specific parameters
        parser.add_argument("--p-field", type=float, help="Field masking probability")
        parser.add_argument("--p-row", type=float, help="Row masking probability")

        # Checkpoint and logging

        # Experiment tracking
        parser.add_argument("--run-name", type=str, help="Run name")
        parser.add_argument("--seed", type=int, help="Random seed")

        # AMP flags (mutually exclusive) sharing the same destination; default None so YAML controls unless overridden
        amp_group = parser.add_mutually_exclusive_group()
        amp_group.add_argument(
            "--use-amp",
            dest="use_amp",
            action="store_true",
            help="Use automatic mixed precision",
        )
        amp_group.add_argument(
            "--no-use-amp",
            dest="use_amp",
            action="store_false",
            help="Do not use automatic mixed precision",
        )
        parser.add_argument(
            "--head-type",
            type=str,
            choices=["mlp", "lstm"],
            help="Head type (mlp or lstm)",
        )
        parser.set_defaults(use_amp=None)

        return parser.parse_args()

    def merge_cli_with_config(
        self, cli_args: argparse.Namespace, config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge CLI arguments with config file, CLI taking precedence."""
        merged = config_dict.copy()
        cli_dict = vars(cli_args)

        # Map CLI args to config structure
        cli_mapping = {
            "model_type": "model.training.model_type",
            "batch_size": "model.training.batch_size",
            "learning_rate": "model.training.learning_rate",
            "total_epochs": "model.training.total_epochs",
            "device": "model.training.device",
            "field_d_model": "model.field_transformer.d_model",
            "field_n_heads": "model.field_transformer.n_heads",
            "field_depth": "model.field_transformer.depth",
            "seq_d_model": "model.sequence_transformer.d_model",
            "seq_n_heads": "model.sequence_transformer.n_heads",
            "seq_depth": "model.sequence_transformer.depth",
            "window": "model.data.window",
            "stride": "model.data.stride",
            "num_bins": "model.data.num_bins",
            "p_field": "model.training.p_field",
            "p_row": "model.training.p_row",
            "run_name": "metrics.run_name",
            "seed": "metrics.seed",
            "use_amp": "model.training.use_amp",
            "head_type": "model.head_type",
        }

        for cli_key, config_path in cli_mapping.items():
            if cli_key in cli_dict and cli_dict[cli_key] is not None:
                # Navigate to the correct nested location
                keys = config_path.split(".")
                current = merged
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = cli_dict[cli_key]

        return merged

    def load_config(self, config_path: str) -> Config:
        """Load and merge configuration from file and CLI."""

        # Parse CLI arguments
        cli_args = self.parse_cli_args()

        # Override config path from CLI if provided
        if cli_args.config:
            config_path = cli_args.config

        # Resolve path relative to config directory
        config_dir = Path(__file__).parent
        config_file_path = config_dir / config_path

        # Load YAML config
        config_dict = self.load_yaml(str(config_file_path))

        # Merge CLI args with config
        merged_config = self.merge_cli_with_config(cli_args, config_dict)

        # Convert to Config object
        self.config = Config.from_dict(merged_config)

        return self.config

    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert config to dictionary for wandb logging."""
        return config.to_dict()
