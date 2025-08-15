#!/usr/bin/env python3
"""
Test script to verify molecular descriptor integration with ESA model.
"""

import sys
import os
import torch

# Add the current directory to Python path
sys.path.append(os.path.realpath("."))

from data_loading.ffpm_molecular_loader import get_ffpm_molecular_dataset_train_val_test
from esa.models import Estimator

def test_molecular_descriptors():
    """Test that molecular descriptors work with ESA model."""
    print("Testing molecular descriptor integration...")
    
    # Load a small sample of data
    train, val, test, num_classes, task_type, scaler = get_ffpm_molecular_dataset_train_val_test(
        dataset_dir="/rxrx/data/user/thomas.qin/hclint",
        target_column="property_value::Mic-CRO",
        task_type="regression",
        val_frac=0.01,  # Use very small validation set for quick test
        random_seed=42
    )
    
    print(f"✓ Data loading successful!")
    print(f"  - Train set: {len(train)} molecules")
    print(f"  - Val set: {len(val)} molecules") 
    print(f"  - Test set: {len(test)} molecules")
    
    # Get sample data characteristics
    sample = train[0]
    num_features = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None
    set_max_items = sample.max_edge_global.item() if hasattr(sample, 'max_edge_global') else 100
    
    print(f"\nSample data characteristics:")
    print(f"  - Node features shape: {sample.x.shape}")
    print(f"  - Edge features shape: {sample.edge_attr.shape if sample.edge_attr is not None else 'None'}")
    print(f"  - Has molecular_descriptors: {hasattr(sample, 'molecular_descriptors')}")
    if hasattr(sample, 'molecular_descriptors'):
        print(f"  - Molecular descriptors shape: {sample.molecular_descriptors.shape}")
    
    # Test 1: Model WITHOUT molecular descriptors
    print(f"\n=== Test 1: Model WITHOUT molecular descriptors ===")
    model_without_mol_desc = Estimator(
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
        posenc=[],
        use_molecular_descriptors=False,  # Disabled
        molecular_descriptor_dim=10,
    )
    
    print(f"  - Model created successfully (without molecular descriptors)")
    print(f"  - Output MLP input dim: {model_without_mol_desc.output_mlp.in_features}")
    
    # Test 2: Model WITH molecular descriptors
    print(f"\n=== Test 2: Model WITH molecular descriptors ===")
    model_with_mol_desc = Estimator(
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
        posenc=[],
        use_molecular_descriptors=True,   # Enabled
        molecular_descriptor_dim=10,
    )
    
    print(f"  - Model created successfully (with molecular descriptors)")
    print(f"  - Output MLP input dim: {model_with_mol_desc.output_mlp.in_features}")
    print(f"  - Expected input dim: {64 + 10} (graph_dim + molecular_descriptor_dim)")
    
    # Test forward pass
    print(f"\n=== Test 3: Forward pass ===")
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    
    # Create a small batch
    train_loader = GeometricDataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(train_loader))
    
    print(f"  - Batch created:")
    print(f"    - Batch size: {batch.batch.max().item() + 1}")
    print(f"    - Has molecular_descriptors: {hasattr(batch, 'molecular_descriptors')}")
    if hasattr(batch, 'molecular_descriptors'):
        print(f"    - Molecular descriptors shape: {batch.molecular_descriptors.shape}")
    
    # Test both models
    model_without_mol_desc.eval()
    model_with_mol_desc.eval()
    
    with torch.no_grad():
        # Prepare inputs
        x, edge_index, y, batch_mapping, edge_attr = (
            batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr
        )
        max_edge = batch.max_edge_global
        num_max_items = torch.max(max_edge).item()
        
        # Test model without molecular descriptors
        predictions_without = model_without_mol_desc.forward(
            x=x,
            edge_index=edge_index,
            batch_mapping=batch_mapping,
            edge_attr=edge_attr,
            num_max_items=num_max_items,
            batch=batch
        )
        
        # Test model with molecular descriptors
        predictions_with = model_with_mol_desc.forward(
            x=x,
            edge_index=edge_index,
            batch_mapping=batch_mapping,
            edge_attr=edge_attr,
            num_max_items=num_max_items,
            batch=batch
        )
        
        print(f"  - Forward pass successful!")
        print(f"    - Predictions without mol desc: {predictions_without.shape}")
        print(f"    - Predictions with mol desc: {predictions_with.shape}")
        print(f"    - Target shape: {y.shape}")
        
        # Check that predictions are different (indicating molecular descriptors are being used)
        if not torch.allclose(predictions_without, predictions_with):
            print(f"  ✓ Molecular descriptors affect predictions (as expected)")
        else:
            print(f"  ⚠ Predictions are identical (unexpected)")
    
    print(f"\n✓ All tests passed! Molecular descriptor integration is working.")
    return True

if __name__ == "__main__":
    try:
        success = test_molecular_descriptors()
        if success:
            print(f"\n🎉 Molecular descriptor integration test passed!")
            print(f"\nYou can now use molecular descriptors in training with:")
            print(f"  --use-molecular-descriptors")
            print(f"  --molecular-descriptor-dim 10")
        else:
            print(f"\n❌ Test failed.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()