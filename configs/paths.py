"""
Path configuration for the transaction transformer project.
"""
from pathlib import Path
from typing import Dict, Any, Optional


class ProjectPaths:
    """
    Centralized path management for the project.
    
    Handles paths for:
    - Data directories (raw, processed, cached)
    - Model checkpoints and artifacts
    - Logs and outputs
    - Configuration files
    """
    
    def __init__(
        self,
        data_dir: str,
        training_type: str = "pretrain",
        checkpoint_filename: Optional[str] = None,
        pretrain_data_filename: Optional[str] = None,
        finetune_data_filename: Optional[str] = None
    ):
        """
        Initialize project paths.
        
        Args:
            data_dir: Root data directory
            training_type: "pretrain" or "finetune"
            checkpoint_filename: Custom checkpoint filename
            pretrain_data_filename: Custom pretraining data filename
            finetune_data_filename: Custom finetuning data filename
        """
        self.data_dir = Path(data_dir)
        self.training_type = training_type
        
        # Data paths
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "datasets"  # Changed from "processed" to "datasets"
        self.cache_dir = self.data_dir / "cache"
        
        # Model paths
        self.models_dir = self.data_dir / "models"
        self.pretrained_models_dir = self.models_dir / "pretrained"
        self.finetuned_models_dir = self.models_dir / "finetuned"
        
        # Checkpoint paths
        if checkpoint_filename:
            self.checkpoint_filename = checkpoint_filename
        else:
            self.checkpoint_filename = f"{training_type}_checkpoint.pt"
        
        self.checkpoint_path = self.models_dir / self.checkpoint_filename
        
        # Data file paths
        if pretrain_data_filename:
            self.pretrain_data_filename = pretrain_data_filename
        else:
            self.pretrain_data_filename = "legitimate_transactions_processed.pt"  # Updated to match actual file
            
        if finetune_data_filename:
            self.finetune_data_filename = finetune_data_filename
        else:
            self.finetune_data_filename = "all_transactions_processed.pt"  # Updated to match actual file
        
        self.pretrain_data_path = self.processed_data_dir / self.pretrain_data_filename
        self.finetune_data_path = self.processed_data_dir / self.finetune_data_filename
        
        # Log and output paths
        self.logs_dir = self.data_dir / "logs"
        self.outputs_dir = self.data_dir / "outputs"
        
        # Configuration paths
        self.configs_dir = Path("configs")
    
    def ensure_dirs_exist(self):
        """Create all necessary directories if they don't exist."""
        dirs_to_create = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.models_dir,
            self.pretrained_models_dir,
            self.finetuned_models_dir,
            self.logs_dir,
            self.outputs_dir
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_custom_checkpoint_path(self, filename: str) -> Path:
        """Get path for a custom checkpoint file."""
        return self.models_dir / filename
    
    def __str__(self) -> str:
        """String representation of all paths."""
        paths_str = f"ProjectPaths (training_type={self.training_type}):\n"
        paths_str += f"  Data directory: {self.data_dir}\n"
        paths_str += f"  Raw data: {self.raw_data_dir}\n"
        paths_str += f"  Processed data: {self.processed_data_dir}\n"
        paths_str += f"  Cache: {self.cache_dir}\n"
        paths_str += f"  Models: {self.models_dir}\n"
        paths_str += f"  Pretrained models: {self.pretrained_models_dir}\n"
        paths_str += f"  Finetuned models: {self.finetuned_models_dir}\n"
        paths_str += f"  Checkpoint: {self.checkpoint_path}\n"
        paths_str += f"  Pretrain data: {self.pretrain_data_path}\n"
        paths_str += f"  Finetune data: {self.finetune_data_path}\n"
        paths_str += f"  Logs: {self.logs_dir}\n"
        paths_str += f"  Outputs: {self.outputs_dir}\n"
        return paths_str


def create_paths_from_config(config: Dict[str, Any]) -> ProjectPaths:
    """
    Create ProjectPaths instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing path-related keys
        
    Returns:
        ProjectPaths instance
    """
    # Extract path-related configuration
    data_dir = config.get("data_dir", "data")
    training_type = config.get("training_type", "pretrain")
    checkpoint_filename = config.get("checkpoint_filename")
    pretrain_data_filename = config.get("pretrain_data_filename")
    finetune_data_filename = config.get("finetune_data_filename")
    
    return ProjectPaths(
        data_dir=data_dir,
        training_type=training_type,
        checkpoint_filename=checkpoint_filename,
        pretrain_data_filename=pretrain_data_filename,
        finetune_data_filename=finetune_data_filename
    )
