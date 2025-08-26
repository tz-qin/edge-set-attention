"""
Clean Training Script for ESA with Proper Multi-Task Support

This script provides a clean training interface that uses the new
multi-task architecture while maintaining backward compatibility.

Usage Examples:

Single-task:
    python -m esa.train_clean \
        --dataset FFPM_MOLECULAR \
        --dataset-download-dir /path/to/data \
        --target-column property_value::Mic-CRO \
        --task-type regression \
        --graph-dim 256 \
        --lr 0.001

Multi-task:
    python -m esa.train_clean \
        --dataset FFPM_MOLECULAR \
        --dataset-download-dir /path/to/data \
        --target-columns property_value::Mic-CRO property_value::Mic-RADME \
        --task-type regression \
        --graph-dim 256 \
        --lr 0.001
"""

import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import torch
import numpy as np

from esa.task_config import create_task_config_from_args, TaskConfiguration
from esa.data_module import create_data_module
from esa.estimators import create_estimator, EstimatorTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for clean training script."""
    parser = argparse.ArgumentParser(
        description="Train ESA models with clean multi-task support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., FFPM_MOLECULAR)")
    parser.add_argument("--dataset-download-dir", type=str, required=True,
                       help="Path to dataset directory or file")
    parser.add_argument("--dataset-dir", type=str,
                       help="Alias for dataset-download-dir (for backward compatibility)")
    parser.add_argument("--dataset-one-hot", action="store_true",
                       help="Use one-hot encoding (legacy compatibility, ignored)")
    
    # Task configuration
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--target-column", type=str,
                           help="Single target column for single-task learning")
    task_group.add_argument("--target-columns", type=str, nargs="+",
                           help="Multiple target columns for multi-task learning")
    
    parser.add_argument("--task-type", type=str, default="regression",
                       choices=["regression", "binary_classification", "multi_classification"],
                       help="Type of task")
    parser.add_argument("--loss-function", type=str, default="mse",
                       choices=["mse", "mae", "bce", "ce"],
                       help="Loss function to use")
    parser.add_argument("--regression-loss-fn", type=str, default="mse",
                       choices=["mse", "mae"],
                       help="Regression loss function (alias for loss-function)")
    parser.add_argument("--task-weights", type=float, nargs="+",
                       help="Task weights for multi-task learning (optional)")
    
    # Model architecture
    parser.add_argument("--graph-dim", type=int, default=256,
                       help="Graph representation dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256],
                       help="Hidden dimensions for attention layers")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 8, 8, 8],
                       help="Number of attention heads per layer")
    parser.add_argument("--layer-types", type=str, nargs="+", default=["M", "S", "M", "P"],
                       help="Layer types (M=Masked, S=Self, P=PMA)")
    parser.add_argument("--apply-attention-on", type=str, default="edge",
                       choices=["edge", "node"],
                       help="Apply attention on edges or nodes")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=200,
                       help="Maximum number of epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=30,
                       help="Early stopping patience")
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-3,
                       help="Optimizer weight decay")
    parser.add_argument("--gradient-clip-val", type=float, default=0.5,
                       help="Gradient clipping value")
    
    # Regularization
    parser.add_argument("--sab-dropout", type=float, default=0.0,
                       help="Self-attention block dropout")
    parser.add_argument("--mab-dropout", type=float, default=0.0,
                       help="Masked attention block dropout")
    parser.add_argument("--pma-dropout", type=float, default=0.0,
                       help="PMA dropout")
    parser.add_argument("--mlp-dropout", type=float, default=0.0,
                       help="MLP dropout")
    
    # MLP configuration
    parser.add_argument("--use-mlps", action="store_true", default=True,
                       help="Use MLPs in attention blocks")
    parser.add_argument("--mlp-hidden-size", type=int, default=128,
                       help="MLP hidden size")
    parser.add_argument("--mlp-layers", type=int, default=3,
                       help="Number of MLP layers")
    parser.add_argument("--mlp-type", type=str, default="standard",
                       choices=["standard", "gated_mlp"],
                       help="Type of MLP")
    parser.add_argument("--use-mlp-ln", action="store_true",
                       help="Use layer normalization in MLPs")
    
    # Architecture options
    parser.add_argument("--norm-type", type=str, default="LN",
                       choices=["BN", "LN"],
                       help="Normalization type")
    parser.add_argument("--pre-or-post", type=str, default="post",
                       choices=["pre", "post"],
                       help="Pre or post layer normalization")
    parser.add_argument("--xformers-or-torch-attn", type=str, default="xformers",
                       choices=["xformers", "torch"],
                       help="Attention implementation")
    parser.add_argument("--use-bfloat16", action="store_true", default=True,
                       help="Use bfloat16 mixed precision")
    
    # Positional encoding
    parser.add_argument("--posenc", type=str,
                       choices=["RWSE", "LapPE", "RWSE+LapPE"],
                       help="Positional encoding type")
    
    # Molecular descriptors
    parser.add_argument("--use-molecular-descriptors", action="store_true",
                       help="Use molecular descriptors")
    parser.add_argument("--molecular-descriptor-dim", type=int, default=10,
                       help="Molecular descriptor dimension")
    
    # Data loading
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loading workers")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                       help="Validation fraction")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (alias for random-seed)")
    
    # Output and monitoring
    parser.add_argument("--out-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--out-path", type=str,
                       help="Output path (alias for out-dir)")
    parser.add_argument("--experiment-name", type=str, default="esa_experiment",
                       help="Experiment name for logging")
    parser.add_argument("--monitor-loss-name", type=str,
                       help="Loss name to monitor for early stopping")
    
    # Hardware
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["32", "16", "bf16"],
                       help="Training precision")
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Handle legacy argument aliases
    if hasattr(args, 'dataset_dir') and args.dataset_dir:
        args.dataset_download_dir = args.dataset_dir
    if hasattr(args, 'regression_loss_fn') and args.regression_loss_fn:
        args.loss_function = args.regression_loss_fn
    if hasattr(args, 'seed') and args.seed != 42:  # Only use if explicitly set
        args.random_seed = args.seed
    if hasattr(args, 'out_path') and args.out_path:
        args.out_dir = args.out_path
    
    # Validate that dimensions match
    if len(args.hidden_dims) != len(args.num_heads):
        raise ValueError(f"hidden_dims ({len(args.hidden_dims)}) and num_heads ({len(args.num_heads)}) must have same length")
    
    if len(args.hidden_dims) != len(args.layer_types):
        raise ValueError(f"hidden_dims ({len(args.hidden_dims)}) and layer_types ({len(args.layer_types)}) must have same length")
    
    # Validate task weights if provided
    if args.target_columns and args.task_weights:
        if len(args.target_columns) != len(args.task_weights):
            raise ValueError(f"Number of task weights ({len(args.task_weights)}) must match number of target columns ({len(args.target_columns)})")
    
    # Validate paths
    if not Path(args.dataset_download_dir).exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_download_dir}")


def setup_logging_and_callbacks(args: argparse.Namespace) -> tuple:
    """Setup logging and training callbacks."""
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger_instance = TensorBoardLogger(
        save_dir=str(output_dir),
        name=args.experiment_name
    )
    
    # Setup callbacks
    callbacks = []
    
    # Use the monitor loss name from args (consistent with original train.py)
    monitor_metric = "val_loss"  # Always use val_loss to avoid dataloader indexing issues
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stopping_patience,
        mode="min" if "loss" in monitor_metric else "max",
        verbose=False  # Disable verbose improvement notifications
    )
    callbacks.append(early_stop_callback)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / args.experiment_name),
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor=monitor_metric,
        mode="min" if "loss" in monitor_metric else "max",
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    return logger_instance, callbacks


def main():
    """Main training function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Set default monitor loss name if not provided (match original train.py)
    if args.monitor_loss_name is None:
        args.monitor_loss_name = "val_loss"
    
    if args.monitor_loss_name == "MCC" or "MCC" in args.monitor_loss_name:
        args.monitor_loss_name = "Validation MCC"
    
    logger.info(f"Starting training with arguments: {args}")
    
    # Set random seed
    pl.seed_everything(args.random_seed)
    
    # Create task configuration
    task_config = create_task_config_from_args(
        target_name=args.target_column,
        target_columns=args.target_columns,
        task_type=args.task_type,
        loss_function=args.loss_function,
        task_weights=args.task_weights
    )
    
    logger.info(f"Task configuration: {task_config}")
    
    # Create data module
    data_module = create_data_module(
        task_config=task_config,
        dataset_dir=args.dataset_download_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_molecular_descriptors=args.use_molecular_descriptors,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed
    )
    
    # Setup data
    data_split, dataset_info = data_module.setup()
    
    logger.info(f"Dataset info: {dataset_info}")
    logger.info(f"Data split: {data_split.num_samples}")
    
    # Create estimator
    estimator = create_estimator(
        task_config=task_config,
        dataset_info=dataset_info,
        graph_dim=args.graph_dim,
        hidden_dims=args.hidden_dims,
        num_heads=args.num_heads,
        layer_types=args.layer_types,
        apply_attention_on=args.apply_attention_on,
        lr=args.lr,
        batch_size=args.batch_size,
        optimiser_weight_decay=args.optimiser_weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.early_stopping_patience,
        sab_dropout=args.sab_dropout,
        mab_dropout=args.mab_dropout,
        pma_dropout=args.pma_dropout,
        use_mlps=args.use_mlps,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_type=args.mlp_type,
        mlp_layers=args.mlp_layers,
        mlp_dropout=args.mlp_dropout,
        use_mlp_ln=args.use_mlp_ln,
        norm_type=args.norm_type,
        pre_or_post=args.pre_or_post,
        xformers_or_torch_attn=args.xformers_or_torch_attn,
        use_bfloat16=args.use_bfloat16,
        posenc=args.posenc,
        use_molecular_descriptors=args.use_molecular_descriptors,
        monitor_loss_name=args.monitor_loss_name
    )
    
    logger.info(f"Created estimator: {type(estimator).__name__}")
    
    # Create trainer helper
    trainer_helper = EstimatorTrainer(
        estimator=estimator,
        data_split=data_split,
        dataset_info=dataset_info
    )
    
    # Validate data compatibility
    trainer_helper.validate_data_compatibility()
    
    # Fit scalers if needed
    trainer_helper.fit_scalers()
    
    # Setup logging and callbacks
    logger_instance, callbacks = setup_logging_and_callbacks(args)
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        precision="bf16" if args.use_bfloat16 else "32",
        callbacks=callbacks,
        logger=logger_instance,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=20,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    logger.info("Starting training...")
    
    # Train model
    trainer.fit(
        model=estimator,
        train_dataloaders=data_split.train,
        val_dataloaders=data_split.val
    )
    
    logger.info("Training completed!")
    
    # Test model
    logger.info("Running final evaluation...")
    test_results = trainer.test(
        model=estimator,
        dataloaders=data_split.test,
        ckpt_path="best"
    )
    
    logger.info(f"Test results: {test_results}")
    
    # Save final model info
    output_dir = Path(args.out_dir) / args.experiment_name
    
    # Save task configuration
    with open(output_dir / "task_config.txt", "w") as f:
        f.write(str(task_config))
    
    # Save dataset info
    with open(output_dir / "dataset_info.txt", "w") as f:
        f.write(f"Dataset Info:\n")
        f.write(f"  Num features: {dataset_info.num_features}\n")
        f.write(f"  Edge dim: {dataset_info.edge_dim}\n")
        f.write(f"  Max nodes: {dataset_info.max_nodes}\n")
        f.write(f"  Max edges: {dataset_info.max_edges}\n")
        f.write(f"  Molecular descriptor dim: {dataset_info.molecular_descriptor_dim}\n")
        f.write(f"  Task column mapping: {dataset_info.task_column_mapping}\n")
        
        # Training and test R² scores are logged automatically via the metric system
    
    logger.info(f"Training completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()