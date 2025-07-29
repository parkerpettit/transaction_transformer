#!/usr/bin/env python
"""
Clean entry point for TransactionModel pretraining.
"""
import argparse
import sys
import time
import signal
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.utils import load_cfg, merge, load_ckpt
from training.data_loader import load_processed_data, create_dataloaders
from training.config_builder import build_model_config, create_model, setup_performance_optimizations
from training.trainer import PretrainTrainer
from training.logging_utils import TrainingLogger
from configs.paths import create_paths_from_config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    ap = argparse.ArgumentParser(description="Train / fine-tune TransactionModel")

    # Paths and run control
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ap.add_argument("--config", type=str, help="YAML config file")
    ap.add_argument("--data_dir", type=str, help="Root directory of processed data")
    
    # Path overrides (optional)
    ap.add_argument("--checkpoint_filename", type=str, help="Custom checkpoint filename")
    ap.add_argument("--pretrain_data_filename", type=str, help="Custom pretraining data filename") 
    ap.add_argument("--finetune_data_filename", type=str, help="Custom finetuning data filename")

    # Training hyperparameters
    ap.add_argument("--total_epochs", type=int, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, help="Batch size for training")
    ap.add_argument("--lr", type=float, help="Initial learning rate")
    ap.add_argument("--window", type=int, help="Sequence length (transactions per sample)")
    ap.add_argument("--stride", type=int, help="Stride length between windows")

    # Feature lists
    ap.add_argument("--cat_features", type=str, nargs="+", help="Categorical column names")
    ap.add_argument("--cont_features", type=str, nargs="+", help="Continuous column names")

    # Architecture: embedding layer
    ap.add_argument("--emb_dropout", type=float, help="Dropout after embedding layer")

    # Field-level transformer
    ap.add_argument("--ft_d_model", type=int, help="Field-transformer hidden dimension")
    ap.add_argument("--ft_depth", type=int, help="Field-transformer number of layers")
    ap.add_argument("--ft_n_heads", type=int, help="Field-transformer number of attention heads")
    ap.add_argument("--ft_ffn_mult", type=int, help="Field-transformer feedforward expansion factor")
    ap.add_argument("--ft_dropout", type=float, help="Dropout within field-transformer")
    ap.add_argument("--ft_layer_norm_eps", type=float, help="Layer norm epsilon for field-transformer")
    ap.add_argument("--ft_norm_first", type=bool, help="Norm first for field-transformer")

    # Sequence-level transformer
    ap.add_argument("--seq_d_model", type=int, help="Sequence-transformer hidden dimension")
    ap.add_argument("--seq_depth", type=int, help="Sequence-transformer number of layers")
    ap.add_argument("--seq_n_heads", type=int, help="Sequence-transformer number of attention heads")
    ap.add_argument("--seq_ffn_mult", type=int, help="Sequence-transformer feedforward expansion factor")
    ap.add_argument("--seq_dropout", type=float, help="Dropout within sequence-transformer")
    ap.add_argument("--seq_layer_norm_eps", type=float, help="Layer norm epsilon for sequence-transformer")
    ap.add_argument("--seq_norm_first", type=bool, help="Norm first for sequence-transformer")

    # Classification layer
    ap.add_argument("--clf_dropout", type=float, help="Dropout before final classification layer")

    # Training mode
    ap.add_argument("--mode", type=str, choices=["ar", "masked", "mlm"],
                    help="Training mode: 'ar' for autoregressive, 'masked'/'mlm' for masked language model")
    ap.add_argument("--mask_prob", type=float, help="Probability of masking tokens in MLM mode")

    # Performance optimizations
    ap.add_argument("--use_mixed_precision", type=bool, help="Enable automatic mixed precision training")
    ap.add_argument("--gradient_checkpointing", type=bool, help="Enable gradient checkpointing to save memory")
    ap.add_argument("--compile_model", type=bool, help="Enable torch.compile for faster training")

    return ap


def setup_graceful_exit():
    """Setup graceful exit handler for Ctrl-C."""
    def graceful_exit(signum=None, frame=None):
        print("[Ctrl-C] Exiting gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, graceful_exit)


def main():
    """Main training function."""
    setup_graceful_exit()
    
    # Parse arguments
    parser = create_argument_parser()
    cli = parser.parse_args()
    
    # Set default config if not provided
    if cli.config is None:
        cli.config = "configs/pretrain.yaml"
    
    # Load and merge configuration
    config_path = cli.config
    if not Path(config_path).is_absolute():
        config_path = project_root / config_path
    
    file_params = load_cfg(config_path)
    args = merge(cli, file_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Set training type for path configuration
        args.training_type = "pretrain"
        
        # Create path configuration
        paths = create_paths_from_config(vars(args))
        paths.ensure_dirs_exist()
        print(f"Using paths:\n{paths}")
        
        # Load processed data
        print("Loading processed data...")
        (train_df, val_df, test_df, enc, cat_features, 
         cont_features, scaler, qparams) = load_processed_data(paths, mode="pretrain")
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            train_df, val_df, cat_features, cont_features,
            args.batch_size, args.window, args.stride,
            mode=args.mode,
            mask_prob=args.mask_prob,
            masking_mode="field"  # Use field-level masking by default
        )
        
        # Setup model and training
        if args.resume:
            print("Resuming from checkpoint...")
            ckpt_path = paths.checkpoint_path
            model, best_val, start_epoch = load_ckpt(ckpt_path, device=device)
            model.to(device)
            config = model.cfg
        else:
            print("Creating new model...")
            config = build_model_config(args, cat_features, cont_features, enc, qparams)
            model = create_model(config, device)
            model = setup_performance_optimizations(model, args)
            start_epoch = 0
            best_val = float("inf")
        
        # Setup logging
        logger = TrainingLogger(
            project_name="txn-transformer",
            run_name=f"pretrain-{Path(args.data_dir).stem}",
            config=vars(args),
            resume=args.resume
        )
        logger.watch_model(model)
        
        # Create trainer
        trainer = PretrainTrainer(
            model=model,
            config=config,
            args=args,
            device=device,
            cat_features=cat_features,
            cont_features=cont_features,
            logger=logger,
            encoders=enc,
            qparams=qparams
        )
        trainer.best_val_loss = best_val
        
        # Training loop
        print("Starting training...")
        ckpt_path = paths.checkpoint_path
        print(f"Stride length: {args.stride}, window: {args.window}, batch_size: {args.batch_size}, total_epochs: {args.total_epochs}, lr: {args.lr}, mode: {args.mode}")
        print(f"Saving checkpoint to: {ckpt_path}, using file path {args.pretrain_data_filename} for pretraining.")
        for epoch in range(start_epoch, args.total_epochs):
            # Train epoch
            t0 = time.perf_counter()
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_loss, feat_acc = trainer.evaluate_model(val_loader, show_predictions=True)
            t1 = time.perf_counter()
            epoch_time = (t1 - t0) / 60
            
            # Log epoch metrics
            logger.log_epoch_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch + 1,
                epoch_time_min=epoch_time,
                learning_rate=trainer.optimizer.param_groups[0]['lr'],
                samples_per_epoch=getattr(train_loader.dataset, '__len__', lambda: 0)(),
                batches_per_epoch=len(train_loader),
                best_val_loss=trainer.best_val_loss,
                epochs_without_improvement=trainer.epochs_without_improvement,
                feature_accuracies=feat_acc
            )
            
            print(f"Epoch {epoch+1:02}/{args.total_epochs} "
                  f"| train {train_loss:.4f}  val {val_loss:.4f} "
                  f"| Time: {epoch_time:.2f} min")
            
            # Check for improvement and save checkpoint
            if not trainer.check_early_stopping(val_loss):
                if trainer.best_val_loss == val_loss:
                    print("New best model! Saving checkpoint...")
                    trainer.save_checkpoint(ckpt_path, epoch)
                    logger.log_best_model(trainer.best_val_loss)
                    print(f"New best ({trainer.best_val_loss:.4f}), checkpoint saved.")
                else:
                    print(f"No improvement for {trainer.patience} epochs. Stopping early.")
                    break
            
        # Save final artifacts
        logger.log_artifact(ckpt_path, paths.checkpoint_filename, "model")
        logger.finish()
        print("Pre-training complete! Backbone saved.")
        
        return args, logger.run
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA OOM at batch_size={args.batch_size}")
            if 'logger' in locals():
                logger.alert("CUDA OOM", f"Run crashed at batch_size={args.batch_size}")
                logger.finish(exit_code=97)
            sys.exit(97)
        else:
            print(f"Runtime error: {e}")
            if 'logger' in locals():
                logger.finish(exit_code=98)
            sys.exit(98)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args, run = main()
    
    if args is None or run is None:
        print("Training failed - data not found. Please run preprocessing first.")
        sys.exit(1)
    
    print("Training completed successfully!") 