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
    is_causal: bool = True  # Only used for sequence transformer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layer."""
    emb_dim: int = 72  # Should match field transformer d_model
    dropout: float = 0.1
    padding_idx: int = 0
    freq_encoding_L: int = 8  # For continuous features
    mask_token_init_std: float = 0.02

@dataclass
class ClassificationConfig:
    """Configuration for classification head."""
    hidden_dim: int = 512
    depth: int = 2  # Number of layers in the MLP. 0 is a linear layer
    dropout: float = 0.1
    output_dim: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training
    total_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1.0e-4
    
    # Training mode
    mode: str = "ar"  # "mlm" or "ar"
    
    # MLM-specific parameters
    p_field: float = 0.15  # Field masking probability
    p_row: float = 0.10    # Row masking probability
    joint_timestamp_masking: bool = True
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "linear", "none"
    mixed_precision: bool = False
    
    # Checkpointing and logging
    early_stopping_patience: int = 5
    
    # Device and workers
    device: str = "cuda"  # "auto", "cpu", "cuda"
    num_workers: int = 4


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Data paths
    data_dir: str = "data"
    preprocessed_path: str = "data/preprocessing/legit_processed.pt"
    
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
    model_type: str = "feature_prediction"  # "feature_prediction", "fraud_detection"
    
    # Transformer configurations
    field_transformer: TransformerConfig = field(default_factory=TransformerConfig)
    
    sequence_transformer: TransformerConfig = field(default_factory=TransformerConfig)
    
    # Classification configuration
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    
    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Row projection
    row_types: int = 1  # Number of different row types
    
    # Model parameters
    freeze_embedding: bool = False
    emb_dropout: float = 0.1
    clf_dropout: float = 0.1
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    checkpoint_dir: str = "data/models"
    log_dir: str = "logs"


@dataclass
class MetricsConfig:
    """Configuration for experiment tracking and logging."""
    run_name: str = "pretrain"
    
    # Logging
    wandb: bool = True
    wandb_project: str = "feature-predictor"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Logging frequency
    log_gradients: bool = False
    log_parameters: bool = False


@dataclass
class Config:
    """Complete configuration for the feature prediction transformer."""
    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate transformer configurations
        if self.model.field_transformer.d_model != self.model.embedding.emb_dim:
            raise ValueError(
                f"Field transformer d_model ({self.model.field_transformer.d_model}) "
                f"must match embedding emb_dim ({self.model.embedding.emb_dim})"
            )
        
        # Validate training mode
        if self.model.training.mode not in ["mlm", "ar"]:
            raise ValueError(f"Training mode must be 'mlm' or 'ar', got {self.model.training.mode}")
        
        # Validate masking probabilities
        if not 0 <= self.model.training.p_field <= 1:
            raise ValueError(f"p_field must be between 0 and 1, got {self.model.training.p_field}")
        if not 0 <= self.model.training.p_row <= 1:
            raise ValueError(f"p_row must be between 0 and 1, got {self.model.training.p_row}")
        
        # Validate device
        if self.model.training.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Device must be 'auto', 'cpu', or 'cuda', got {self.model.training.device}")
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Handle nested configs
        if 'model' in config_dict:
            model_dict = config_dict['model']
            if 'field_transformer' in model_dict:
                model_dict['field_transformer'] = TransformerConfig(**model_dict['field_transformer'])
            if 'sequence_transformer' in model_dict:
                model_dict['sequence_transformer'] = TransformerConfig(**model_dict['sequence_transformer'])
            if 'embedding' in model_dict:
                model_dict['embedding'] = EmbeddingConfig(**model_dict['embedding'])
            if 'training' in model_dict:
                model_dict['training'] = TrainingConfig(**model_dict['training'])
            if 'data' in model_dict:
                model_dict['data'] = DataConfig(**model_dict['data'])
            if 'classification' in model_dict:
                model_dict['classification'] = ClassificationConfig(**model_dict['classification'])
            
            config_dict['model'] = ModelConfig(**model_dict)
        
        if 'metrics' in config_dict:
            config_dict['metrics'] = MetricsConfig(**config_dict['metrics'])
        
        return cls(**config_dict)


class ConfigManager:
    """Manages configuration loading, merging, and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[Config] = None
    
    def load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    
    def parse_cli_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Transaction Transformer Training")
        
        # Configuration file
        parser.add_argument("--config", type=str, default="pretrain.yaml",
                          help="Path to configuration file")
        
        # Model parameters
        parser.add_argument("--model-type", type=str, help="Type of model to train")
        parser.add_argument("--training-mode", type=str, choices=["mlm", "ar"], 
                          help="Pretraining mode (MLM or autoregressive)")
        
        # Training parameters
        parser.add_argument("--batch-size", type=int, help="Batch size")
        parser.add_argument("--learning-rate", type=float, help="Learning rate")
        parser.add_argument("--total-epochs", type=int, help="Total training epochs")
        parser.add_argument("--device", type=str, help="Device to use (cpu, cuda, auto)")
        
        # Transformer parameters
        parser.add_argument("--field-d-model", type=int, help="Field transformer d_model")
        parser.add_argument("--field-n-heads", type=int, help="Field transformer number of heads")
        parser.add_argument("--field-depth", type=int, help="Field transformer depth")
        parser.add_argument("--seq-d-model", type=int, help="Sequence transformer d_model")
        parser.add_argument("--seq-n-heads", type=int, help="Sequence transformer number of heads")
        parser.add_argument("--seq-depth", type=int, help="Sequence transformer depth")
        
        # Data parameters
        parser.add_argument("--data-dir", type=str, help="Data directory")
        parser.add_argument("--window", type=int, help="Sequence window size")
        parser.add_argument("--stride", type=int, help="Stride between windows")
        parser.add_argument("--num-bins", type=int, help="Number of bins for continuous features")
        
        # MLM-specific parameters
        parser.add_argument("--p-field", type=float, help="Field masking probability")
        parser.add_argument("--p-row", type=float, help="Row masking probability")
        
        # Checkpoint and logging
        parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
        parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
        parser.add_argument("--log-dir", type=str, help="Logging directory")
        
        # Experiment tracking
        parser.add_argument("--run-name", type=str, help="Run name")
        parser.add_argument("--seed", type=int, help="Random seed")
        
        return parser.parse_args()
    
    def merge_cli_with_config(self, cli_args: argparse.Namespace, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Merge CLI arguments with config file, CLI taking precedence."""
        merged = config_dict.copy()
        cli_dict = vars(cli_args)
        
        # Map CLI args to config structure
        cli_mapping = {
            'model_type': 'model.model_type',
            'training_mode': 'model.training.mode',
            'batch_size': 'model.training.batch_size',
            'learning_rate': 'model.training.learning_rate',
            'total_epochs': 'model.training.total_epochs',
            'device': 'model.training.device',
            'field_d_model': 'model.field_transformer.d_model',
            'field_n_heads': 'model.field_transformer.n_heads',
            'field_depth': 'model.field_transformer.depth',
            'seq_d_model': 'model.sequence_transformer.d_model',
            'seq_n_heads': 'model.sequence_transformer.n_heads',
            'seq_depth': 'model.sequence_transformer.depth',
            'data_dir': 'model.data.data_dir',
            'window': 'model.data.window',
            'stride': 'model.data.stride',
            'num_bins': 'model.data.num_bins',
            'p_field': 'model.training.p_field',
            'p_row': 'model.training.p_row',
            'checkpoint_dir': 'model.checkpoint_dir',
            'log_dir': 'model.log_dir',
            'run_name': 'metrics.run_name',
            'seed': 'metrics.seed'
        }
        
        for cli_key, config_path in cli_mapping.items():
            if cli_key in cli_dict and cli_dict[cli_key] is not None:
                # Navigate to the correct nested location
                keys = config_path.split('.')
                current = merged
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = cli_dict[cli_key]
        
        return merged
    
    def load_config(self, config_path: Optional[str] = None) -> Config:
        """Load and merge configuration from file and CLI."""
        if config_path is None:
            config_path = self.config_path or "pretrain.yaml"
        
        # Parse CLI arguments
        cli_args = self.parse_cli_args()
        
        # Override config path from CLI if provided
        if cli_args.config:
            config_path = cli_args.config
        
        # Ensure config_path is not None
        if config_path is None:
            config_path = "pretrain.yaml"
        
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
    
    def get_config(self) -> Config:
        """Get the current configuration."""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert config to dictionary for wandb logging."""
        return config.to_dict() 