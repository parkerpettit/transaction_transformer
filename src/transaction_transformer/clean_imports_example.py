"""
Example demonstrating the clean import structure.

All imports are now minimal and sexy!
"""

# Super clean imports from the main package
from transaction_transformer import (
    # Data preprocessing
    FieldSchema, preprocess,
    
    # Datasets and collators  
    TxnDataset, MLMTabCollator, ARTabCollator,
    
    # Models
    FeaturePredictionModel, TransactionEmbeddingModel, FraudDetectionModel,
    
    # Config
    ModelConfig, TransformerConfig
)

# Import trainers from their specific modules
from transaction_transformer.modeling.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from transaction_transformer.modeling.training.trainers.mlm_trainer import MLMTrainer
from transaction_transformer.modeling.training.base.base_trainer import BaseTrainer

# Or import specific modules for more granular control
from transaction_transformer.data import TxnDataset
from transaction_transformer.modeling.models import FeaturePredictionModel
from transaction_transformer.modeling.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from transaction_transformer.utils import ModelConfig

# Example usage
def example_usage():
    """Show how clean the imports are now."""
    
    # Create config
    from transaction_transformer.utils.config import NextTransactionPredictionConfig
    
    config = ModelConfig(
        cat_vocab_sizes={"merchant": 100, "category": 50},
        cont_vocab_sizes={"amount": 20},
        ft_config=TransformerConfig(d_model=64),
        seq_config=TransformerConfig(d_model=64),
        next_trans_config=NextTransactionPredictionConfig(),
        lstm_config=None
    )
    
    # Create model
    model = FeaturePredictionModel(config)
    
    # Create trainer (commented out since it requires actual data)
    # trainer = AutoregressiveTrainer(
    #     model=model,
    #     schema=FieldSchema(),  # This would be loaded from data
    #     config=config,
    #     device="cpu",
    #     train_loader=None,  # This would be created from dataset
    #     val_loader=None,
    #     optimizer=None,
    #     scheduler=None
    # )
    
    print("Clean imports working perfectly!")
    print(f"Model: {type(model)}")
    print(f"Config: {type(config)}")

if __name__ == "__main__":
    example_usage() 