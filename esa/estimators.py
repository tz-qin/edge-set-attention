"""
Single-Task and Multi-Task Estimators for ESA

This module provides clean implementations of single-task and multi-task
estimators that inherit from the base estimator.

Design Principles:
- Clear separation between single and multi-task logic
- Composition over inheritance for task heads
- Clean, testable implementations
- Proper error handling and validation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
import logging

from esa.base_estimator import BaseEstimator
from esa.task_config import TaskConfiguration, TaskType
from esa.task_heads import TaskHead, TaskHeadCollection, create_task_head
from esa.data_module import DatasetInfo

logger = logging.getLogger(__name__)


class SingleTaskEstimator(BaseEstimator):
    """
    Clean single-task estimator implementation.
    
    Handles single regression or classification tasks with a simple,
    straightforward implementation.
    """
    
    def __init__(
        self,
        task_config: TaskConfiguration,
        dataset_info: DatasetInfo,
        **kwargs
    ):
        """
        Initialize single-task estimator.
        
        Args:
            task_config: Task configuration (must be single-task)
            dataset_info: Dataset information
            **kwargs: Additional arguments passed to base class
        """
        if task_config.is_multi_task:
            raise ValueError("SingleTaskEstimator requires single-task configuration")
        
        super().__init__(task_config, dataset_info, **kwargs)
        
        logger.info(f"Initialized SingleTaskEstimator for task: {self.task_config.tasks[0].name}")
    
    def _setup_task_heads(self) -> None:
        """Setup single task head."""
        task_def = self.task_config.tasks[0]
        
        self.task_head = create_task_head(
            task_definition=task_def,
            input_dim=self.final_feature_dim,
            hidden_size=self.mlp_hidden_size,
            num_layers=self.mlp_layers,
            dropout=self.mlp_dropout,
            use_layer_norm=self.use_mlp_ln
        )
        
        logger.info(f"Created single task head: {self.task_head}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Any
    ) -> torch.Tensor:
        """
        Forward pass for single-task prediction.
        
        Returns:
            Predictions tensor [batch_size, output_dim]
        """
        # Get graph representations from ESA
        graph_representations = self._forward_esa(
            x, edge_index, batch_mapping, edge_attr, batch
        )
        
        # Get final features (with optional molecular descriptors)
        final_features = self._get_final_features(graph_representations, batch)
        
        # Predict using task head
        predictions = self.task_head(final_features)
        
        return predictions.flatten()
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch: Any
    ) -> torch.Tensor:
        """Compute single-task loss."""
        # Ensure targets are in correct format
        if targets.dim() > 1:
            targets = targets.flatten()
        
        return self.task_head.compute_loss(predictions, targets)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch: Any,
        step_type: str = "train"
    ) -> Dict[str, float]:
        """Compute single-task metrics."""
        # Ensure targets are in correct format
        if targets.dim() > 1:
            targets = targets.flatten()
        
        metrics = self.task_head.compute_metrics(predictions, targets)
        
        # Add task name prefix to avoid confusion
        task_name = self.task_config.tasks[0].name
        return {f"{task_name}_{k}": v for k, v in metrics.items()}
    
    def fit_scaler(self, targets: torch.Tensor) -> None:
        """Fit scaler for regression tasks."""
        if self.task_config.global_task_type == TaskType.REGRESSION:
            targets_np = targets.detach().cpu().numpy()
            if targets_np.ndim > 1:
                targets_np = targets_np.flatten()
            self.task_head.fit_scaler(targets_np)
    
    def unscale_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Unscale predictions to original scale."""
        return self.task_head.unscale_predictions(predictions)


class MultiTaskEstimator(BaseEstimator):
    """
    Clean multi-task estimator implementation.
    
    Manages multiple task heads and handles multi-task loss computation
    and metrics aggregation with proper error handling.
    """
    
    def __init__(
        self,
        task_config: TaskConfiguration,
        dataset_info: DatasetInfo,
        **kwargs
    ):
        """
        Initialize multi-task estimator.
        
        Args:
            task_config: Task configuration (must be multi-task)
            dataset_info: Dataset information
            **kwargs: Additional arguments passed to base class
        """
        if not task_config.is_multi_task:
            raise ValueError("MultiTaskEstimator requires multi-task configuration")
        
        super().__init__(task_config, dataset_info, **kwargs)
        
        logger.info(f"Initialized MultiTaskEstimator for {len(self.task_config.tasks)} tasks")
    
    def _setup_task_heads(self) -> None:
        """Setup multiple task heads using TaskHeadCollection."""
        task_heads = {}
        
        for task_def in self.task_config.tasks:
            task_head = create_task_head(
                task_definition=task_def,
                input_dim=self.final_feature_dim,
                hidden_size=self.mlp_hidden_size,
                num_layers=self.mlp_layers,
                dropout=self.mlp_dropout,
                use_layer_norm=self.use_mlp_ln
            )
            task_heads[task_def.name] = task_head
        
        self.task_head_collection = TaskHeadCollection(task_heads)
        
        logger.info(f"Created {len(task_heads)} task heads: {list(task_heads.keys())}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task prediction.
        
        Returns:
            Dictionary mapping task names to predictions
        """
        # Get graph representations from ESA
        graph_representations = self._forward_esa(
            x, edge_index, batch_mapping, edge_attr, batch
        )
        
        # Get final features (with optional molecular descriptors)
        final_features = self._get_final_features(graph_representations, batch)
        
        # Predict using all task heads
        predictions = self.task_head_collection(final_features)
        
        # Flatten predictions for each task
        for task_name in predictions:
            predictions[task_name] = predictions[task_name].flatten()
        
        return predictions
    
    def _compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: Any
    ) -> torch.Tensor:
        """Compute multi-task loss with proper weighting."""
        # Handle PyTorch Geometric batching: targets are flattened as [batch_size * num_tasks]
        # Need to reshape to [batch_size, num_tasks]
        expected_num_tasks = len(self.task_config.tasks)
        batch_size = batch.batch_size
        
        if targets.dim() == 1 and targets.shape[0] == batch_size * expected_num_tasks:
            # Reshape from PyG flattened format to [batch_size, num_tasks]
            targets = targets.view(batch_size, expected_num_tasks)
        elif targets.dim() == 1:
            # Single column case - need to expand
            targets = targets.unsqueeze(1)
        
        # Validate target dimensions
        if targets.shape[1] != expected_num_tasks:
            raise ValueError(
                f"Target tensor has {targets.shape[1]} columns but expected {expected_num_tasks} "
                f"for tasks: {self.task_config.task_names}"
            )
        
        # Compute weighted loss across all tasks
        total_loss, task_losses = self.task_head_collection.compute_losses(
            predictions=predictions,
            targets=targets,
            task_column_mapping=self.dataset_info.task_column_mapping,
            weights=self.task_config.task_weights
        )
        
        # Don't log individual task losses to keep output clean
        # Individual losses are still computed for the weighted total loss
        
        return total_loss
    
    def _compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: Any,
        step_type: str = "train"
    ) -> Dict[str, float]:
        """Compute multi-task metrics."""
        # Handle PyTorch Geometric batching: targets are flattened as [batch_size * num_tasks]
        # Need to reshape to [batch_size, num_tasks]
        expected_num_tasks = len(self.task_config.tasks)
        batch_size = batch.batch_size
        
        if targets.dim() == 1 and targets.shape[0] == batch_size * expected_num_tasks:
            # Reshape from PyG flattened format to [batch_size, num_tasks]
            targets = targets.view(batch_size, expected_num_tasks)
        elif targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Compute metrics for all tasks
        all_metrics = self.task_head_collection.compute_metrics(
            predictions=predictions,
            targets=targets,
            task_column_mapping=self.dataset_info.task_column_mapping
        )
        
        # Flatten metrics dictionary (include all metrics like single task)
        flat_metrics = {}
        for task_name, task_metrics in all_metrics.items():
            for metric_name, metric_value in task_metrics.items():
                flat_metrics[f"{task_name}_{metric_name}"] = metric_value
        
        # For training: only return R² to match single task behavior  
        if step_type == "train":
            train_r2_metrics = {k: v for k, v in flat_metrics.items() if "_r2" in k}
            return train_r2_metrics
        
        # For test: only return relevant metrics (filter out tasks not in test set)
        if step_type == "test":
            # Only keep Mic-CRO metrics since test set only contains that data
            test_metrics = {}
            for k, v in flat_metrics.items():
                if "Mic-CRO" in k and ("_r2" in k or "_mse" in k):
                    test_metrics[k] = v
            return test_metrics
        
        # For validation: return all metrics
        return flat_metrics
    
    def fit_scalers(self, targets: torch.Tensor) -> None:
        """Fit scalers for all regression task heads."""
        if self.task_config.global_task_type == TaskType.REGRESSION:
            # Handle PyTorch Geometric batching for multi-task scalers
            expected_num_tasks = len(self.task_config.tasks)
            
            # For multi-task, need to handle the flattened format from PyG batching
            if targets.dim() == 1 and targets.shape[0] % expected_num_tasks == 0:
                # Reshape from flattened format to [num_samples, num_tasks]
                num_samples = targets.shape[0] // expected_num_tasks
                targets = targets.view(num_samples, expected_num_tasks)
            elif targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            self.task_head_collection.fit_scalers(
                targets=targets,
                task_column_mapping=self.dataset_info.task_column_mapping
            )
    
    def unscale_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unscale predictions to original scale for all tasks."""
        unscaled_predictions = {}
        
        for task_name, task_predictions in predictions.items():
            if task_name in self.task_head_collection.task_heads:
                task_head = self.task_head_collection.task_heads[task_name]
                unscaled_predictions[task_name] = task_head.unscale_predictions(task_predictions)
            else:
                unscaled_predictions[task_name] = task_predictions
        
        return unscaled_predictions
    
    @property
    def task_heads(self) -> Dict[str, TaskHead]:
        """Get dictionary of task heads."""
        return dict(self.task_head_collection.task_heads)


def create_estimator(
    task_config: TaskConfiguration,
    dataset_info: DatasetInfo,
    **kwargs
) -> BaseEstimator:
    """
    Factory function to create appropriate estimator based on task configuration.
    
    Args:
        task_config: Validated task configuration
        dataset_info: Dataset information
        **kwargs: Additional arguments passed to estimator
        
    Returns:
        Appropriate estimator instance (SingleTaskEstimator or MultiTaskEstimator)
        
    Raises:
        ValueError: If task configuration is invalid
    """
    if task_config.is_multi_task:
        return MultiTaskEstimator(
            task_config=task_config,
            dataset_info=dataset_info,
            **kwargs
        )
    else:
        return SingleTaskEstimator(
            task_config=task_config,
            dataset_info=dataset_info,
            **kwargs
        )


class EstimatorTrainer:
    """
    High-level trainer class that encapsulates the entire training pipeline.
    
    Provides a clean interface for training estimators with proper
    data handling, scaler fitting, and validation.
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        data_split: Any,  # DataSplit from data_module
        dataset_info: DatasetInfo
    ):
        """
        Initialize trainer.
        
        Args:
            estimator: Trained estimator (SingleTaskEstimator or MultiTaskEstimator)
            data_split: Data split with train/val/test loaders
            dataset_info: Dataset information
        """
        self.estimator = estimator
        self.data_split = data_split
        self.dataset_info = dataset_info
        
        logger.info(f"Initialized EstimatorTrainer with {type(estimator).__name__}")
    
    def fit_scalers(self) -> None:
        """Fit scalers on training data for regression tasks."""
        if self.estimator.task_config.global_task_type != TaskType.REGRESSION:
            logger.info("Skipping scaler fitting for non-regression tasks")
            return
        
        logger.info("Fitting scalers on training data...")
        
        # Collect all training targets
        all_targets = []
        for batch in self.data_split.train:
            all_targets.append(batch.y)
        
        targets = torch.cat(all_targets, dim=0)
        
        # Fit scalers based on estimator type
        if isinstance(self.estimator, SingleTaskEstimator):
            self.estimator.fit_scaler(targets)
        elif isinstance(self.estimator, MultiTaskEstimator):
            self.estimator.fit_scalers(targets)
        
        logger.info("Scaler fitting complete")
    
    def validate_data_compatibility(self) -> None:
        """Validate that data is compatible with estimator configuration."""
        # Get sample batch
        sample_batch = next(iter(self.data_split.train))
        
        # Check target dimensions
        targets = sample_batch.y
        if targets is None:
            raise ValueError("Sample batch has no targets (y is None)")
            
        expected_tasks = len(self.estimator.task_config.tasks)
        batch_size = sample_batch.batch_size
        
        if self.estimator.task_config.is_multi_task:
            # For multi-task, PyTorch Geometric flattens targets to [batch_size * num_tasks]
            # We need to ensure the total size is divisible by num_tasks
            if targets.dim() != 1:
                raise ValueError(
                    f"Multi-task estimator expects flattened targets from PyG batching, "
                    f"got shape {targets.shape} with {targets.dim()} dimensions"
                )
            
            expected_total_size = batch_size * expected_tasks
            if targets.shape[0] != expected_total_size:
                raise ValueError(
                    f"Multi-task estimator expects {expected_total_size} total targets "
                    f"({batch_size} batch_size * {expected_tasks} tasks), "
                    f"got {targets.shape[0]} targets"
                )
        else:
            # Single task can handle both [batch_size] and [batch_size, 1] shapes
            if targets.shape[0] != batch_size:
                raise ValueError(
                    f"Single-task estimator expects {batch_size} targets for batch size, "
                    f"got {targets.shape[0]} targets"
                )
        
        # Check feature dimensions
        expected_features = self.dataset_info.num_features
        actual_features = sample_batch.x.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} node features, got {actual_features}"
            )
        
        # Check edge dimensions if applicable
        if sample_batch.edge_attr is not None:
            expected_edge_dim = self.dataset_info.edge_dim
            actual_edge_dim = sample_batch.edge_attr.shape[1]
            if actual_edge_dim != expected_edge_dim:
                raise ValueError(
                    f"Expected {expected_edge_dim} edge features, got {actual_edge_dim}"
                )
        
        logger.info("Data compatibility validation passed")


def setup_estimator_pipeline(
    task_config: TaskConfiguration,
    dataset_dir: str,
    batch_size: int = 32,
    **estimator_kwargs
) -> EstimatorTrainer:
    """
    Convenience function to setup the complete estimator training pipeline.
    
    Args:
        task_config: Task configuration
        dataset_dir: Dataset directory or file path
        batch_size: Batch size for data loaders
        **estimator_kwargs: Additional arguments for estimator
        
    Returns:
        Configured EstimatorTrainer ready for training
        
    Raises:
        ValueError: If configuration or data is invalid
    """
    from esa.data_module import load_data_for_task_config
    
    # Load data
    data_split, dataset_info = load_data_for_task_config(
        task_config=task_config,
        dataset_dir=dataset_dir,
        batch_size=batch_size
    )
    
    # Create estimator
    estimator = create_estimator(
        task_config=task_config,
        dataset_info=dataset_info,
        **estimator_kwargs
    )
    
    # Create trainer
    trainer = EstimatorTrainer(
        estimator=estimator,
        data_split=data_split,
        dataset_info=dataset_info
    )
    
    # Validate compatibility
    trainer.validate_data_compatibility()
    
    # Fit scalers if needed
    trainer.fit_scalers()
    
    logger.info("Estimator pipeline setup complete")
    
    return trainer