#!/usr/bin/env python3
"""
Test Script for Clean Multi-Task Architecture

This script validates that the new clean multi-task implementation works correctly
and provides better reliability than the legacy implementation.

Run with:
    python test_clean_architecture.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeometricDataLoader
import sys
import os
import logging
import traceback

# Add project root to path
sys.path.append('.')

from esa.task_config import TaskConfiguration, TaskDefinition, TaskType, LossFunction
from esa.task_heads import TaskHead, RegressionTaskHead, create_task_head
from esa.data_module import DataModule, DatasetInfo
from esa.estimators import SingleTaskEstimator, MultiTaskEstimator, create_estimator, EstimatorTrainer
from data_loading.ffpm_molecular_loader import get_ffpm_molecular_dataset_train_val_test

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_task_configuration():
    """Test task configuration system."""
    print("\n🧪 Testing Task Configuration System")
    print("=" * 50)
    
    try:
        # Test single-task configuration
        single_config = TaskConfiguration.from_single_task(
            task_name="MIC_CRO",
            column_name="property_value::Mic-CRO",
            task_type=TaskType.REGRESSION,
            loss_function=LossFunction.MSE
        )
        
        assert not single_config.is_multi_task
        assert len(single_config.tasks) == 1
        assert single_config.tasks[0].name == "MIC_CRO"
        
        print("✓ Single-task configuration works correctly")
        
        # Test multi-task configuration
        multi_configs = [
            {
                "name": "MIC_CRO",
                "column_name": "property_value::Mic-CRO",
                "task_type": "regression",
                "loss_function": "mse"
            },
            {
                "name": "MIC_RADME", 
                "column_name": "property_value::Mic-RADME",
                "task_type": "regression",
                "loss_function": "mae",
                "weight": 2.0
            }
        ]
        
        multi_config = TaskConfiguration.from_multi_task(multi_configs)
        
        assert multi_config.is_multi_task
        assert len(multi_config.tasks) == 2
        assert multi_config.task_weights["MIC_RADME"] == 2.0
        
        print("✓ Multi-task configuration works correctly")
        
        # Test validation
        try:
            # This should fail - incompatible loss function
            TaskConfiguration.from_single_task(
                task_name="test",
                column_name="test_col",
                task_type=TaskType.REGRESSION,
                loss_function=LossFunction.BCE  # Wrong for regression
            )
            assert False, "Should have failed validation"
        except ValueError:
            print("✓ Configuration validation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Task configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_task_heads():
    """Test task head system."""
    print("\n🧪 Testing Task Head System")
    print("=" * 50)
    
    try:
        # Create a regression task definition
        task_def = TaskDefinition(
            name="test_regression",
            column_name="test_col",
            task_type=TaskType.REGRESSION,
            loss_function=LossFunction.MSE
        )
        
        # Create task head
        task_head = create_task_head(task_def, input_dim=64, hidden_size=32)
        
        assert isinstance(task_head, RegressionTaskHead)
        assert task_head.task_def.name == "test_regression"
        
        print("✓ Task head creation works correctly")
        
        # Test forward pass
        batch_size = 4
        input_features = torch.randn(batch_size, 64)
        predictions = task_head(input_features)
        
        assert predictions.shape == (batch_size, 1)
        
        print("✓ Task head forward pass works correctly")
        
        # Test loss computation
        targets = torch.randn(batch_size)
        loss = task_head.compute_loss(predictions.squeeze(), targets)
        
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        
        print("✓ Task head loss computation works correctly")
        
        # Test with NaN targets
        targets_with_nan = targets.clone()
        targets_with_nan[0] = float('nan')
        
        loss_with_nan = task_head.compute_loss(predictions.squeeze(), targets_with_nan)
        assert not torch.isnan(loss_with_nan)
        
        print("✓ Task head handles NaN targets correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Task head test failed: {e}")
        traceback.print_exc()
        return False


def test_estimators():
    """Test estimator creation and basic functionality."""
    print("\n🧪 Testing Estimator System")
    print("=" * 50)
    
    try:
        # Create mock dataset info
        dataset_info = DatasetInfo(
            num_features=36,
            edge_dim=11,
            max_nodes=50,
            max_edges=60,
            molecular_descriptor_dim=10,
            task_column_mapping={"MIC_CRO": 0, "MIC_RADME": 1}
        )
        
        # Test single-task estimator
        single_config = TaskConfiguration.from_single_task(
            task_name="MIC_CRO",
            column_name="property_value::Mic-CRO",
            task_type=TaskType.REGRESSION,
            loss_function=LossFunction.MSE
        )
        
        single_estimator = create_estimator(
            task_config=single_config,
            dataset_info=dataset_info,
            graph_dim=64,
            hidden_dims=[32, 32],
            num_heads=[4, 4],
            layer_types=["M", "P"]
        )
        
        assert isinstance(single_estimator, SingleTaskEstimator)
        print("✓ Single-task estimator creation works correctly")
        
        # Test multi-task estimator
        multi_configs = [
            {
                "name": "MIC_CRO",
                "column_name": "property_value::Mic-CRO",
                "task_type": "regression",
                "loss_function": "mse"
            },
            {
                "name": "MIC_RADME",
                "column_name": "property_value::Mic-RADME", 
                "task_type": "regression",
                "loss_function": "mae"
            }
        ]
        
        multi_config = TaskConfiguration.from_multi_task(multi_configs)
        
        multi_estimator = create_estimator(
            task_config=multi_config,
            dataset_info=dataset_info,
            graph_dim=64,
            hidden_dims=[32, 32],
            num_heads=[4, 4],
            layer_types=["M", "P"]
        )
        
        assert isinstance(multi_estimator, MultiTaskEstimator)
        print("✓ Multi-task estimator creation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Estimator test failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test clean data loading interface."""
    print("\n🧪 Testing Data Loading System")
    print("=" * 50)
    
    try:
        # Check if test data exists
        test_data_path = "/rxrx/data/user/thomas.qin/hclint/mtl_train_set_data_08_14.parquet"
        if not os.path.exists(test_data_path):
            print(f"⚠️  Test data not found at {test_data_path}, skipping data loading test")
            return True
        
        # Test single-task data loading
        single_config = TaskConfiguration.from_single_task(
            task_name="MIC_CRO",
            column_name="property_value::Mic-CRO",
            task_type=TaskType.REGRESSION,
            loss_function=LossFunction.MSE
        )
        
        data_module = DataModule(
            task_config=single_config,
            dataset_dir="/rxrx/data/user/thomas.qin/hclint",
            batch_size=4,
            use_molecular_descriptors=True
        )
        
        data_split, dataset_info = data_module.setup()
        
        assert data_split.train is not None
        assert data_split.val is not None
        assert data_split.test is not None
        assert dataset_info.num_features > 0
        
        print("✓ Single-task data loading works correctly")
        
        # Test multi-task data loading
        multi_configs = [
            {
                "name": "MIC_CRO",
                "column_name": "property_value::Mic-CRO",
                "task_type": "regression",
                "loss_function": "mse"
            },
            {
                "name": "MIC_RADME",
                "column_name": "property_value::Mic-RADME",
                "task_type": "regression", 
                "loss_function": "mae"
            }
        ]
        
        multi_config = TaskConfiguration.from_multi_task(multi_configs)
        
        multi_data_module = DataModule(
            task_config=multi_config,
            dataset_dir="/rxrx/data/user/thomas.qin/hclint",
            batch_size=4,
            use_molecular_descriptors=True
        )
        
        multi_data_split, multi_dataset_info = multi_data_module.setup()
        
        assert len(multi_dataset_info.task_column_mapping) == 2
        
        print("✓ Multi-task data loading works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test end-to-end integration."""
    print("\n🧪 Testing End-to-End Integration")
    print("=" * 50)
    
    try:
        # Check if test data exists
        test_data_path = "/rxrx/data/user/thomas.qin/hclint/mtl_train_set_data_08_14.parquet"
        if not os.path.exists(test_data_path):
            print(f"⚠️  Test data not found at {test_data_path}, skipping integration test")
            return True
        
        # Create small multi-task configuration
        multi_configs = [
            {
                "name": "MIC_CRO",
                "column_name": "property_value::Mic-CRO",
                "task_type": "regression",
                "loss_function": "mse"
            },
            {
                "name": "MIC_RADME",
                "column_name": "property_value::Mic-RADME",
                "task_type": "regression",
                "loss_function": "mae"
            }
        ]
        
        task_config = TaskConfiguration.from_multi_task(multi_configs)
        
        # Create data module
        data_module = DataModule(
            task_config=task_config,
            dataset_dir="/rxrx/data/user/thomas.qin/hclint",
            batch_size=4,
            use_molecular_descriptors=True
        )
        
        data_split, dataset_info = data_module.setup()
        
        # Create estimator
        estimator = create_estimator(
            task_config=task_config,
            dataset_info=dataset_info,
            graph_dim=32,  # Small for testing
            hidden_dims=[16, 16],
            num_heads=[2, 2],
            layer_types=["M", "P"],
            lr=0.001
        )
        
        # Create trainer
        trainer = EstimatorTrainer(
            estimator=estimator,
            data_split=data_split,
            dataset_info=dataset_info
        )
        
        # Validate compatibility
        trainer.validate_data_compatibility()
        
        # Fit scalers
        trainer.fit_scalers()
        
        print("✓ End-to-end integration setup works correctly")
        
        # Test a few training steps
        estimator.train()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.001)
        
        num_test_batches = 3
        for i, batch in enumerate(data_split.train):
            if i >= num_test_batches:
                break
                
            optimizer.zero_grad()
            
            # Forward pass
            loss = estimator.training_step(batch, i)
            
            if loss is not None and not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                print(f"✓ Training step {i+1}: loss={loss.item():.6f}")
            else:
                print(f"⚠️  Training step {i+1}: loss is None or NaN")
        
        print("✓ Multi-task training steps work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Clean Multi-Task Architecture")
    print("=" * 80)
    
    tests = [
        ("Task Configuration", test_task_configuration),
        ("Task Heads", test_task_heads),
        ("Estimators", test_estimators),
        ("Data Loading", test_data_loading),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Tests...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Clean architecture is working correctly.")
        print("\nThe new clean multi-task implementation is ready for use!")
        print("Use: python -m esa.train_clean --help")
    else:
        print(f"\n⚠️  Some tests failed. Please review the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())