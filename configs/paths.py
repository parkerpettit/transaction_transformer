"""
Centralized path management for the transaction transformer project.
Provides a single source of truth for all file paths with configurable overrides.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class ProjectPaths:
    """Centralized path configuration for the project."""
    
    # Base directories  
    data_dir: Union[str, Path]
    project_root: Path = Path(__file__).parent.parent
    training_type: str = "pretrain"  # "pretrain" or "finetune"
    
    # Configurable filenames (can be overridden)
    checkpoint_filename: str = "pretrained_backbone.pt"
    pretrain_data_filename: str = "legitimate_transactions_processed.pt"
    finetune_data_filename: str = "all_transactions_processed.pt"
    
    def __post_init__(self):
        """Convert string paths to Path objects and ensure they're absolute."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
            
        # Make paths absolute
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()
    
    @property
    def datasets_dir(self) -> Path:
        """Directory containing processed datasets."""
        return Path(self.data_dir) / "datasets"
    
    @property
    def checkpoints_dir(self) -> Path:
        """Directory containing model checkpoints."""
        return Path(self.data_dir) / "models"
    
    @property
    def pretrained_models_dir(self) -> Path:
        """Directory containing pretrained model checkpoints."""
        return self.checkpoints_dir / "pretrained"
    
    @property
    def finetuned_models_dir(self) -> Path:
        """Directory containing finetuned model checkpoints."""
        return self.checkpoints_dir / "finetuned"
    
    @property
    def pretrain_data_path(self) -> Path:
        """Path to pretraining dataset."""
        if self.pretrain_data_filename is None:
            raise ValueError("pretrain_data_filename is not set")
        return self.datasets_dir / self.pretrain_data_filename
    
    @property
    def finetune_data_path(self) -> Path:
        """Path to finetuning dataset."""
        if self.finetune_data_filename is None:
            raise ValueError("finetune_data_filename is not set")
        return self.datasets_dir / self.finetune_data_filename
    
    @property
    def checkpoint_path(self) -> Path:
        """Path to model checkpoint based on training type."""
        if self.checkpoint_filename is None:
            raise ValueError("checkpoint_filename is not set")
        
        if self.training_type == "pretrain":
            return self.pretrained_models_dir / self.checkpoint_filename
        elif self.training_type == "finetune":
            return self.finetuned_models_dir / self.checkpoint_filename
        else:
            raise ValueError(f"Unknown training_type: {self.training_type}. Must be 'pretrain' or 'finetune'")
    
    def get_custom_checkpoint_path(self, filename: str) -> Path:
        """Get path for a custom checkpoint filename based on training type."""
        if self.training_type == "pretrain":
            return self.pretrained_models_dir / filename
        elif self.training_type == "finetune":
            return self.finetuned_models_dir / filename
        else:
            raise ValueError(f"Unknown training_type: {self.training_type}. Must be 'pretrain' or 'finetune'")
    
    def get_custom_data_path(self, filename: str) -> Path:
        """Get path for a custom data filename."""
        return self.datasets_dir / filename
    
    def ensure_dirs_exist(self):
        """Create directories if they don't exist."""
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_models_dir.mkdir(parents=True, exist_ok=True)
        self.finetuned_models_dir.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation showing all paths."""
        return (
            f"ProjectPaths:\n"
            f"  project_root: {self.project_root}\n"
            f"  data_dir: {self.data_dir}\n"
            f"  training_type: {self.training_type}\n"
            f"  datasets_dir: {self.datasets_dir}\n"
            f"  checkpoints_dir: {self.checkpoints_dir}\n"
            f"  pretrained_models_dir: {self.pretrained_models_dir}\n"
            f"  finetuned_models_dir: {self.finetuned_models_dir}\n"
            f"  pretrain_data: {self.pretrain_data_path}\n"
            f"  finetune_data: {self.finetune_data_path}\n"
            f"  checkpoint: {self.checkpoint_path}"
        )


def create_paths_from_config(config: Dict[str, Any]) -> ProjectPaths:
    """
    Create ProjectPaths from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing path settings
        
    Returns:
        ProjectPaths instance
    """
    # Extract base data directory (required)
    data_dir = config.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir must be specified in configuration")
    
    # Extract training type (default to pretrain)
    training_type = config.get("training_type", "pretrain")
    if training_type not in ["pretrain", "finetune"]:
        raise ValueError(f"training_type must be 'pretrain' or 'finetune', got: {training_type}")
    
    # Extract optional path overrides, filtering out None values
    checkpoint_filename = config.get("checkpoint_filename")
    if checkpoint_filename is None:
        checkpoint_filename = "big_legit_backbone.pt"
        
    pretrain_data_filename = config.get("pretrain_data_filename")
    if pretrain_data_filename is None:
        pretrain_data_filename = "legitimate_transactions_processed.pt"
        
    finetune_data_filename = config.get("finetune_data_filename")
    if finetune_data_filename is None:
        finetune_data_filename = "all_transactions_processed.pt"
    
    # Extract optional path overrides
    paths = ProjectPaths(
        data_dir=data_dir,
        training_type=training_type,
        checkpoint_filename=checkpoint_filename,
        pretrain_data_filename=pretrain_data_filename,
        finetune_data_filename=finetune_data_filename
    )
    
    return paths


def get_default_paths(data_dir: str, training_type: str = "pretrain") -> ProjectPaths:
    """
    Get default paths for a given data directory.
    
    Args:
        data_dir: Base data directory
        training_type: Type of training ("pretrain" or "finetune")
        
    Returns:
        ProjectPaths with default filenames
    """
    if training_type not in ["pretrain", "finetune"]:
        raise ValueError(f"training_type must be 'pretrain' or 'finetune', got: {training_type}")
    
    return ProjectPaths(data_dir=data_dir, training_type=training_type)
