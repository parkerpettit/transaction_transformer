"""
Example usage of transaction_transformer with clean imports.

This demonstrates how to use the package with the new simplified import structure.
"""

# Clean imports - import entire modules
from transaction_transformer.data import (
    TxnDataset, 
    MLMTabCollator, 
    ARTabCollator, 
    FieldSchema,
    preprocess,
    get_encoders,
    get_scaler
)

from transaction_transformer.modeling.models import (
    FeaturePredictionModel,
    TransactionEmbeddingModel,
    FraudDetectionModel
)

from transaction_transformer.modeling.training import (
    create_model_config,
    create_datasets
)

# Import trainers from their specific modules
from transaction_transformer.modeling.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from transaction_transformer.modeling.training.trainers.mlm_trainer import MLMTrainer

from transaction_transformer.utils import (
    ModelConfig,
    TransformerConfig
)

# Or import everything from the main package
from transaction_transformer import (
    FieldSchema,
    TxnDataset,
    FeaturePredictionModel,
    ModelConfig,
    create_model_config
)

def example_usage():
    """Example of how to use the package with clean imports."""
    
    # Create a simple schema
    schema = FieldSchema(
        cat_features=["User", "Card", "Merchant"],
        cont_features=["Amount"],
        cat_encoders={},
        cont_binners={},
        time_cat=["Hour"],
        scaler=None
    )
    
    # Create model config
    from transaction_transformer.utils.config import NextTransactionPredictionConfig
    
    config = ModelConfig(
        cat_vocab_sizes={"User": 100, "Card": 50, "Merchant": 200},
        cont_vocab_sizes={"Amount": 10},
        ft_config=TransformerConfig(d_model=72, n_heads=8, depth=2),
        seq_config=TransformerConfig(d_model=72, n_heads=4, depth=2),
        emb_dropout=0.1,
        clf_dropout=0.1,
        row_types=1,
        freeze_embedding=False,
        next_trans_config=NextTransactionPredictionConfig(),
        lstm_config=None
    )
    
    # Create model
    model = FeaturePredictionModel(config)
    
    print("Successfully imported and created objects with clean imports!")
    print(f"Schema: {schema}")
    print(f"Model: {model}")

if __name__ == "__main__":
    example_usage() 