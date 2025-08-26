"""
Unified Data Module for ESA Single and Multi-Task Learning

This module provides a clean, unified interface for data loading that handles
both single-task and multi-task scenarios transparently.

Design Principles:
- Single interface for all data loading scenarios
- Clean separation between data loading and model logic
- Proper validation and error handling
- Extensible for new data sources
"""

from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

from esa.task_config import TaskConfiguration, TaskDefinition, TaskType
from data_loading.ffpm_molecular_loader import (
    FFPMMolecularDataset, 
    FFPMMolecularDatasetMultiTask
)
from data_loading.transforms import AddMaxEdge, AddMaxNode, FormatSingleLabel
from data_loading.data_loading import CustomPyGDataset
import torch_geometric.transforms as T

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    train: GeometricDataLoader
    val: GeometricDataLoader
    test: GeometricDataLoader
    num_samples: Dict[str, int]
    
    def __post_init__(self):
        """Validate data split after creation."""
        if not all([self.train, self.val, self.test]):
            raise ValueError("All data splits (train, val, test) must be provided")
            
        self.num_samples = {
            "train": len(self.train.dataset),
            "val": len(self.val.dataset), 
            "test": len(self.test.dataset)
        }


@dataclass
class DatasetInfo:
    """Container for dataset metadata and statistics."""
    num_features: int
    edge_dim: int
    max_nodes: int
    max_edges: int
    molecular_descriptor_dim: int
    task_column_mapping: Dict[str, int]  # Maps task names to column indices
    
    def __post_init__(self):
        """Validate dataset info after creation."""
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}")
        if self.edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {self.edge_dim}")


class DataModule:
    """
    Unified data module for ESA training.
    
    Handles data loading, preprocessing, and validation for both single-task
    and multi-task learning scenarios with a clean, consistent interface.
    """
    
    def __init__(
        self,
        task_config: TaskConfiguration,
        dataset_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        use_molecular_descriptors: bool = False,
        val_fraction: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize data module.
        
        Args:
            task_config: Task configuration defining targets and types
            dataset_dir: Path to dataset directory or file
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            use_molecular_descriptors: Whether to include molecular descriptors
            val_fraction: Fraction of training data to use for validation
            random_seed: Random seed for reproducible splits
        """
        self.task_config = task_config
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_molecular_descriptors = use_molecular_descriptors
        self.val_fraction = val_fraction
        self.random_seed = random_seed
        
        # Will be populated during setup
        self.dataset_info: Optional[DatasetInfo] = None
        self.scalers: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized DataModule for {task_config}")
    
    def setup(self) -> Tuple[DataSplit, DatasetInfo]:
        """
        Setup data loading and return data splits and dataset info.
        
        Returns:
            Tuple of (DataSplit, DatasetInfo)
            
        Raises:
            FileNotFoundError: If dataset files are not found
            ValueError: If data loading fails or data is invalid
        """
        logger.info("Setting up data loading...")
        
        # Load raw datasets
        train_dataset, val_dataset, test_dataset = self._load_datasets()
        
        # Extract dataset information
        self.dataset_info = self._extract_dataset_info(train_dataset)
        
        # Create data loaders
        train_loader = self._create_data_loader(train_dataset, shuffle=True)
        val_loader = self._create_data_loader(val_dataset, shuffle=False)
        test_loader = self._create_data_loader(test_dataset, shuffle=False)
        
        data_split = DataSplit(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            num_samples={}  # Will be populated in __post_init__
        )
        
        logger.info(f"Data setup complete: {data_split.num_samples}")
        return data_split, self.dataset_info
    
    def _load_datasets(self) -> Tuple[CustomPyGDataset, CustomPyGDataset, CustomPyGDataset]:
        """
        Load train/validation/test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Check if we have pre-defined splits or need to create them
        if self.dataset_dir.is_file():
            # Single file - create random splits
            return self._load_single_file_with_splits()
        else:
            # Directory - look for pre-defined splits
            return self._load_predefined_splits()
    
    def _load_single_file_with_splits(self) -> Tuple[CustomPyGDataset, CustomPyGDataset, CustomPyGDataset]:
        """Load data from single file and create random splits."""
        logger.info(f"Loading single file: {self.dataset_dir}")
        
        # Set up transforms
        transforms = [AddMaxEdge(), AddMaxNode()]
        # Only add FormatSingleLabel for single-task regression
        if self.task_config.global_task_type == TaskType.REGRESSION and not self.task_config.is_multi_task:
            transforms.append(FormatSingleLabel())
        
        # Create dataset based on task configuration
        if self.task_config.is_multi_task:
            dataset = FFPMMolecularDatasetMultiTask(
                parquet_path=str(self.dataset_dir),
                target_columns=self.task_config.column_names,
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
        else:
            from data_loading.ffpm_molecular_loader import FFPMMolecularDataset
            dataset = FFPMMolecularDataset(
                parquet_path=str(self.dataset_dir),
                target_column=self.task_config.column_names[0],
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
        
        # Create random splits
        train_data, val_data, test_data = self._create_random_splits(dataset)
        
        # Calculate global statistics and apply global transforms
        all_data = train_data + val_data + test_data
        max_nodes = max([data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0 for data in all_data])
        max_edges = max([data.edge_index.shape[1] if hasattr(data, 'edge_index') and data.edge_index is not None else 0 for data in all_data])
        
        # Apply global transforms
        from data_loading.transforms import AddMaxEdgeGlobal, AddMaxNodeGlobal
        global_transform = T.Compose([
            AddMaxEdgeGlobal(max_edges),
            AddMaxNodeGlobal(max_nodes)
        ])
        
        train_data = [global_transform(data) for data in train_data]
        val_data = [global_transform(data) for data in val_data]
        test_data = [global_transform(data) for data in test_data]
        
        return (
            CustomPyGDataset(train_data),
            CustomPyGDataset(val_data), 
            CustomPyGDataset(test_data)
        )
    
    def _load_predefined_splits(self) -> Tuple[CustomPyGDataset, CustomPyGDataset, CustomPyGDataset]:
        """Load data from predefined train/test split files."""
        logger.info(f"Loading predefined splits from directory: {self.dataset_dir}")
        
        # Define file patterns based on task configuration
        if self.task_config.is_multi_task:
            train_file = self.dataset_dir / "mtl_train_set_data_08_14.parquet"
        else:
            train_file = self.dataset_dir / "stl_train_set_08_10.parquet"
        
        test_file = self.dataset_dir / "test_set_08_10.parquet"
        
        # Validate files exist
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        logger.info(f"Found training file: {train_file}")
        logger.info(f"Found test file: {test_file}")
        
        # Set up transforms
        transforms = [AddMaxEdge(), AddMaxNode()]
        # Only add FormatSingleLabel for single-task regression
        if self.task_config.global_task_type == TaskType.REGRESSION and not self.task_config.is_multi_task:
            transforms.append(FormatSingleLabel())
        
        # Load datasets
        if self.task_config.is_multi_task:
            train_dataset = FFPMMolecularDatasetMultiTask(
                parquet_path=str(train_file),
                target_columns=self.task_config.column_names,
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
            
            test_dataset = FFPMMolecularDatasetMultiTask(
                parquet_path=str(test_file),
                target_columns=self.task_config.column_names,
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
        else:
            from data_loading.ffpm_molecular_loader import FFPMMolecularDataset
            train_dataset = FFPMMolecularDataset(
                parquet_path=str(train_file),
                target_column=self.task_config.column_names[0],
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
            
            test_dataset = FFPMMolecularDataset(
                parquet_path=str(test_file),
                target_column=self.task_config.column_names[0],
                task_type=self.task_config.global_task_type.value,
                pre_transform=T.Compose(transforms)
            )
        
        # Create validation split from training data
        train_data, val_data = self._split_training_data(train_dataset)
        
        # Calculate global statistics and apply global transforms
        all_data = train_data + val_data + list(test_dataset)
        max_nodes = max([data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0 for data in all_data])
        max_edges = max([data.edge_index.shape[1] if hasattr(data, 'edge_index') and data.edge_index is not None else 0 for data in all_data])
        
        # Apply global transforms
        from data_loading.transforms import AddMaxEdgeGlobal, AddMaxNodeGlobal
        global_transform = T.Compose([
            AddMaxEdgeGlobal(max_edges),
            AddMaxNodeGlobal(max_nodes)
        ])
        
        train_data = [global_transform(data) for data in train_data]
        val_data = [global_transform(data) for data in val_data]
        test_data = [global_transform(data) for data in list(test_dataset)]
        
        return (
            CustomPyGDataset(train_data),
            CustomPyGDataset(val_data),
            CustomPyGDataset(test_data)
        )
    
    def _create_random_splits(self, dataset) -> Tuple[List[Data], List[Data], List[Data]]:
        """Create random train/val/test splits from dataset."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        data_list = list(dataset)
        n_total = len(data_list)
        
        # Calculate split sizes
        n_test = int(0.15 * n_total)
        n_val = int(0.15 * n_total)
        n_train = n_total - n_test - n_val
        
        # Shuffle data
        indices = np.random.permutation(n_total)
        
        train_data = [data_list[i] for i in indices[:n_train]]
        val_data = [data_list[i] for i in indices[n_train:n_train + n_val]]
        test_data = [data_list[i] for i in indices[n_train + n_val:]]
        
        logger.info(f"Created random splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _split_training_data(self, train_dataset) -> Tuple[List[Data], List[Data]]:
        """Split training dataset into train and validation sets."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        data_list = list(train_dataset)
        n_total = len(data_list)
        n_val = int(self.val_fraction * n_total)
        n_train = n_total - n_val
        
        # Shuffle and split
        indices = np.random.permutation(n_total)
        
        train_data = [data_list[i] for i in indices[:n_train]]
        val_data = [data_list[i] for i in indices[n_train:]]
        
        logger.info(f"Split training data: train={len(train_data)}, val={len(val_data)}")
        
        return train_data, val_data
    
    def _extract_dataset_info(self, dataset) -> DatasetInfo:
        """Extract dataset information from loaded dataset."""
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Get sample data point
        sample = dataset[0]
        
        # Extract dimensions
        num_features = sample.x.shape[1] if sample.x is not None else 0
        edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        
        # Calculate global statistics
        max_nodes = 0
        max_edges = 0
        molecular_descriptor_dim = 0
        
        for data in dataset:
            max_nodes = max(max_nodes, data.x.shape[0])
            if data.edge_attr is not None:
                max_edges = max(max_edges, data.edge_attr.shape[0])
            
            # Check for molecular descriptors
            if hasattr(data, 'molecular_descriptors') and data.molecular_descriptors is not None:
                molecular_descriptor_dim = data.molecular_descriptors.shape[0]
        
        # Create task column mapping
        task_column_mapping = {}
        if self.task_config.is_multi_task:
            for i, task in enumerate(self.task_config.tasks):
                task_column_mapping[task.name] = i
        else:
            task_column_mapping[self.task_config.tasks[0].name] = 0
        
        dataset_info = DatasetInfo(
            num_features=num_features,
            edge_dim=edge_dim,
            max_nodes=max_nodes,
            max_edges=max_edges,
            molecular_descriptor_dim=molecular_descriptor_dim,
            task_column_mapping=task_column_mapping
        )
        
        logger.info(f"Dataset info: {dataset_info}")
        return dataset_info
    
    def _create_data_loader(self, dataset, shuffle: bool = False) -> GeometricDataLoader:
        """Create PyTorch Geometric data loader."""
        return GeometricDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,  # Reduce memory pressure
            persistent_workers=False if self.num_workers > 0 else False
        )
    
    @property
    def input_dim(self) -> int:
        """Get total input dimension including molecular descriptors."""
        if self.dataset_info is None:
            raise RuntimeError("Must call setup() before accessing input_dim")
        
        base_dim = self.dataset_info.num_features
        if self.use_molecular_descriptors:
            base_dim += self.dataset_info.molecular_descriptor_dim
        
        return base_dim


def create_data_module(
    task_config: TaskConfiguration,
    dataset_dir: str,
    batch_size: int = 32,
    **kwargs
) -> DataModule:
    """
    Factory function to create DataModule with validated configuration.
    
    Args:
        task_config: Validated task configuration
        dataset_dir: Path to dataset directory or file
        batch_size: Batch size for data loaders
        **kwargs: Additional arguments passed to DataModule
        
    Returns:
        Configured DataModule instance
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
        ValueError: If configuration is invalid
    """
    # Validate dataset path
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")
    
    # Create and return data module
    return DataModule(
        task_config=task_config,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        **kwargs
    )


def load_data_for_task_config(
    task_config: TaskConfiguration,
    dataset_dir: str,
    batch_size: int = 32,
    **kwargs
) -> Tuple[DataSplit, DatasetInfo]:
    """
    Convenience function to load data with task configuration.
    
    Args:
        task_config: Task configuration
        dataset_dir: Dataset directory or file path
        batch_size: Batch size for data loaders
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (DataSplit, DatasetInfo)
    """
    data_module = create_data_module(task_config, dataset_dir, batch_size, **kwargs)
    return data_module.setup()