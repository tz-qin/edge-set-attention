#!/usr/bin/env python3
"""
Example Usage of Clean Multi-Task Architecture

This script demonstrates how to use the new clean multi-task interface
for both single-task and multi-task learning scenarios.
"""

import sys
import os

# Add project root to path
sys.path.append('.')

from esa.task_config import TaskConfiguration, create_task_config_from_args
from esa.estimators import setup_estimator_pipeline
from esa.train_clean import main as train_clean
import torch
import pytorch_lightning as pl


def example_single_task():
    """Example of single-task learning with clean interface."""
    print("🔬 Single-Task Learning Example")
    print("=" * 50)
    
    # Method 1: Using TaskConfiguration directly
    task_config = TaskConfiguration.from_single_task(
        task_name="MIC_CRO",
        column_name="property_value::Mic-CRO",
        task_type="regression",
        loss_function="mse"
    )
    
    print(f"Task Configuration: {task_config}")
    
    # Method 2: Using command-line style arguments
    task_config_2 = create_task_config_from_args(
        target_name="property_value::Mic-CRO",
        task_type="regression",
        loss_function="mse"
    )
    
    print(f"Task Configuration (CLI style): {task_config_2}")
    
    print("✅ Single-task configuration examples complete")


def example_multi_task():
    """Example of multi-task learning with clean interface."""
    print("\n🔬 Multi-Task Learning Example")
    print("=" * 50)
    
    # Method 1: Using TaskConfiguration directly
    task_configs = [
        {
            "name": "MIC_CRO",
            "column_name": "property_value::Mic-CRO",
            "task_type": "regression",
            "loss_function": "mse",
            "weight": 1.0
        },
        {
            "name": "MIC_RADME",
            "column_name": "property_value::Mic-RADME",
            "task_type": "regression",
            "loss_function": "mae",
            "weight": 2.0
        }
    ]
    
    task_config = TaskConfiguration.from_multi_task(task_configs)
    
    print(f"Multi-Task Configuration: {task_config}")
    print(f"Task weights: {task_config.task_weights}")
    
    # Method 2: Using command-line style arguments
    task_config_2 = create_task_config_from_args(
        target_columns=["property_value::Mic-CRO", "property_value::Mic-RADME"],
        task_type="regression",
        loss_function="mse",
        task_weights=[1.0, 2.0]
    )
    
    print(f"Multi-Task Configuration (CLI style): {task_config_2}")
    
    print("✅ Multi-task configuration examples complete")


def example_training_pipeline():
    """Example of complete training pipeline setup."""
    print("\n🔬 Training Pipeline Example")
    print("=" * 50)
    
    # Check if test data exists
    test_data_path = "/rxrx/data/user/thomas.qin/hclint"
    if not os.path.exists(test_data_path):
        print(f"⚠️  Test data not found at {test_data_path}, skipping pipeline example")
        return
    
    try:
        # Create multi-task configuration
        task_config = create_task_config_from_args(
            target_columns=["property_value::Mic-CRO", "property_value::Mic-RADME"],
            task_type="regression",
            loss_function="mse"
        )
        
        # Setup complete training pipeline
        trainer = setup_estimator_pipeline(
            task_config=task_config,
            dataset_dir=test_data_path,
            batch_size=8,  # Small for example
            graph_dim=64,  # Small for example
            hidden_dims=[32, 32],
            num_heads=[4, 4],
            layer_types=["M", "P"],
            lr=0.001,
            use_molecular_descriptors=True
        )
        
        print(f"✅ Training pipeline setup complete")
        print(f"   Estimator type: {type(trainer.estimator).__name__}")
        print(f"   Data splits: {trainer.data_split.num_samples}")
        print(f"   Dataset info: {trainer.dataset_info.num_features} features")
        
    except Exception as e:
        print(f"⚠️  Pipeline example failed: {e}")


def example_cli_commands():
    """Example CLI commands for training."""
    print("\n🔬 CLI Command Examples")
    print("=" * 50)
    
    print("Single-task training:")
    print("""
python -m esa.train_clean \\
    --dataset FFPM_MOLECULAR \\
    --dataset-download-dir /path/to/data \\
    --target-column property_value::Mic-CRO \\
    --task-type regression \\
    --loss-function mse \\
    --graph-dim 256 \\
    --lr 0.001 \\
    --batch-size 32 \\
    --max-epochs 100
    """)
    
    print("\nMulti-task training:")
    print("""
python -m esa.train_clean \\
    --dataset FFPM_MOLECULAR \\
    --dataset-download-dir /path/to/data \\
    --target-columns property_value::Mic-CRO property_value::Mic-RADME \\
    --task-type regression \\
    --loss-function mse \\
    --task-weights 1.0 2.0 \\
    --graph-dim 256 \\
    --lr 0.001 \\
    --batch-size 32 \\
    --max-epochs 100 \\
    --use-molecular-descriptors
    """)
    
    print("✅ CLI command examples complete")


def main():
    """Run all examples."""
    print("🚀 Clean Multi-Task Architecture Examples")
    print("=" * 80)
    
    example_single_task()
    example_multi_task()
    example_training_pipeline()
    example_cli_commands()
    
    print("\n" + "=" * 80)
    print("🎉 Examples Complete!")
    print("\nNext steps:")
    print("1. For single-task: python -m esa.train_clean --target-column your_column")
    print("2. For multi-task: python -m esa.train_clean --target-columns col1 col2")
    print("3. Run tests: python test_clean_architecture.py")
    print("4. See full options: python -m esa.train_clean --help")


if __name__ == "__main__":
    main()