#!/usr/bin/env python3
"""
Test script for HCLINT molecular data integration with ESA.

This script tests the data loading pipeline and validates that HCLINT
molecular data can be properly loaded and processed for ESA training.
"""

import sys
import os
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Add the current directory to Python path
sys.path.append(os.path.realpath("."))

from data_loading.data_loading import get_dataset_train_val_test


def test_hclint_data_loading():
    """Test loading HCLINT data with ESA-compatible format."""
    print("Testing HCLINT data loading...")
    
    # Path to your featurized parquet file
    parquet_path = "/rxrx/data/user/thomas.qin/hclint/dataset_feated.parquet"
    target_column = "property_value::Mic-CRO"  # Use correct column name
    
    # Test parameters
    dataset_name = "FFPM_MOLECULAR"
    
    try:
        # Load data using ESA's interface
        train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
            dataset=dataset_name,
            dataset_dir=parquet_path,  # For HCLINT, this is the parquet file path
            target_name=target_column,
            task_type="regression",
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            include_molecular_descriptors=True,
            random_seed=42
        )
        
        print(f"✓ Data loading successful!")
        print(f"  - Train set: {len(train)} molecules")
        print(f"  - Val set: {len(val)} molecules") 
        print(f"  - Test set: {len(test)} molecules")
        print(f"  - Task type: {task_type}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Scaler: {'Yes' if scaler else 'No'}")
        
        # Test data sample
        sample = train[0]
        print(f"\nSample data characteristics:")
        print(f"  - Node features shape: {sample.x.shape}")
        print(f"  - Edge features shape: {sample.edge_attr.shape if sample.edge_attr is not None else 'None'}")
        print(f"  - Number of nodes: {sample.num_nodes}")
        print(f"  - Number of edges: {sample.edge_index.shape[1]}")
        print(f"  - Target value: {sample.y}")
        print(f"  - Max node global: {sample.max_node_global}")
        print(f"  - Max edge global: {sample.max_edge_global}")
        
        # Test DataLoader compatibility
        print(f"\nTesting DataLoader compatibility...")
        train_loader = GeometricDataLoader(train, batch_size=4, shuffle=True)
        batch = next(iter(train_loader))
        
        print(f"  - Batch x shape: {batch.x.shape}")
        print(f"  - Batch edge_index shape: {batch.edge_index.shape}")
        print(f"  - Batch edge_attr shape: {batch.edge_attr.shape if batch.edge_attr is not None else 'None'}")
        print(f"  - Batch y shape: {batch.y.shape}")
        print(f"  - Batch size: {batch.batch.max().item() + 1}")
        
        print(f"\n✓ All tests passed! HCLINT data is ESA-compatible.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test that the loaded data is compatible with ESA model forward pass."""
    print(f"\nTesting model compatibility...")
    
    try:
        from esa.models import Estimator
        
        # Load a small sample of data
        parquet_path = "/rxrx/data/user/thomas.qin/hclint/dataset_feated.parquet"
        target_column = "property_value::Mic-CRO"
        
        train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
            dataset="FFPM_MOLECULAR",
            dataset_dir=parquet_path,
            target_name=target_column,
            task_type="regression",
            train_frac=0.01,  # Use only 1% for quick test
            val_frac=0.005,
            test_frac=0.005,
            include_molecular_descriptors=True,
            random_seed=42
        )
        
        # Get sample data characteristics
        sample = train[0]
        num_features = sample.x.shape[1]
        edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None
        set_max_items = sample.max_edge_global if hasattr(sample, 'max_edge_global') else 100
        
        print(f"  - Creating ESA model with:")
        print(f"    - num_features: {num_features}")
        print(f"    - edge_dim: {edge_dim}")
        print(f"    - graph_dim: 64")
        print(f"    - set_max_items: {set_max_items}")
        
        # Create a small ESA model for testing
        model = Estimator(
            task_type=task_type,
            num_features=num_features,
            edge_dim=edge_dim,
            graph_dim=64,
            hidden_dims=[64, 64],
            num_heads=[4, 4],
            layer_types=["M", "P"],
            apply_attention_on="edge",
            linear_output_size=num_classes,
            set_max_items=set_max_items,
            batch_size=2,
            lr=0.001,
            use_mlps=True,
            mlp_hidden_size=32,
            posenc=[],  # Use empty list instead of None
        )
        
        print(f"  - Model created successfully")
        
        # Test forward pass
        train_loader = GeometricDataLoader(train, batch_size=2, shuffle=False)
        batch = next(iter(train_loader))
        
        model.eval()
        with torch.no_grad():
            # Test model forward pass
            x, edge_index, y, batch_mapping, edge_attr = (
                batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr
            )
            max_edge = batch.max_edge_global
            num_max_items = torch.max(max_edge).item()
            
            predictions = model.forward(
                x=x,
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                edge_attr=edge_attr,
                num_max_items=num_max_items,
                batch=batch
            )
            
            print(f"  - Forward pass successful!")
            print(f"  - Input batch size: {batch_mapping.max().item() + 1}")
            print(f"  - Predictions shape: {predictions.shape}")
            print(f"  - Target shape: {y.shape}")
            
        print(f"\n✓ Model compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("HCLINT-ESA Integration Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Data loading
    success &= test_hclint_data_loading()
    
    # Test 2: Model compatibility
    success &= test_model_compatibility()
    
    print(f"\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! HCLINT data is ready for ESA training.")
        print("\nTo use HCLINT data with ESA training, run:")
        print("python -m esa.train \\")
        print("  --dataset FFPM_MOLECULAR \\")
        print("  --dataset-download-dir /rxrx/data/user/thomas.qin/hclint/dataset_feated.parquet \\")
        print("  --dataset-target-name property_value::Mic-CRO \\")
        print("  --regression-loss-fn mse \\")
        print("  --lr 0.001 \\")
        print("  --batch-size 32 \\")
        print("  --graph-dim 128 \\")
        print("  --apply-attention-on edge \\")
        print("  --layer-types M S M P \\")
        print("  --hidden-dims 128 128 128 128 \\")
        print("  --num-heads 8 8 8 8 \\")
        print("  --out-path molecular_esa_output")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)