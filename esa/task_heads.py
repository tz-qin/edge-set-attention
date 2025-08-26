"""
Task Head System for ESA Multi-Task Learning

This module provides modular, reusable task heads that encapsulate
prediction logic, loss computation, and metrics for individual tasks.

Design Principles:
- Single responsibility: each head handles one task
- Composable: heads can be combined for multi-task learning
- Extensible: easy to add new task types
- Testable: clear interfaces and separation of concerns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np

from esa.task_config import TaskDefinition, TaskType, LossFunction
from esa.mlp_utils import SmallMLP

logger = logging.getLogger(__name__)


class TaskHead(ABC, nn.Module):
    """
    Abstract base class for task-specific prediction heads.
    
    Each task head encapsulates:
    - Prediction network (MLP)
    - Loss computation
    - Target scaling/unscaling
    - Task-specific metrics
    """
    
    def __init__(
        self,
        task_definition: TaskDefinition,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.task_def = task_definition
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Create prediction network
        self.prediction_network = SmallMLP(
            in_dim=input_dim,
            inter_dim=hidden_size,
            out_dim=task_definition.output_dim,
            use_ln=use_layer_norm,
            dropout_p=dropout,
            num_layers=num_layers
        )
        
        # Target scaler for this task
        self.scaler: Optional[StandardScaler] = None
        self._is_scaler_fitted = False
        
        logger.info(f"Created {self.__class__.__name__} for task '{task_definition.name}'")
    
    @abstractmethod
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim] 
            mask: Optional mask for valid targets [batch_size]
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim]
            mask: Optional mask for valid targets [batch_size]
            
        Returns:
            Dictionary of metric name to value
        """
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prediction network.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Task predictions [batch_size, output_dim]
        """
        return self.prediction_network(features)
    
    def fit_scaler(self, targets: np.ndarray) -> None:
        """
        Fit scaler on training targets.
        
        Args:
            targets: Training targets [num_samples, output_dim]
        """
        if self.task_def.task_type != TaskType.REGRESSION:
            logger.info(f"Skipping scaler fitting for non-regression task '{self.task_def.name}'")
            return
            
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(targets).any(axis=1) if targets.ndim > 1 else ~np.isnan(targets)
        if not np.any(valid_mask):
            logger.warning(f"No valid targets found for task '{self.task_def.name}', skipping scaler fitting")
            return
            
        valid_targets = targets[valid_mask]
        if valid_targets.ndim == 1:
            valid_targets = valid_targets.reshape(-1, 1)
            
        self.scaler = StandardScaler()
        self.scaler.fit(valid_targets)
        self._is_scaler_fitted = True
        
        logger.info(f"Fitted scaler for task '{self.task_def.name}': mean={self.scaler.mean_[0]:.4f}, std={self.scaler.scale_[0]:.4f}")
    
    def scale_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Scale targets using fitted scaler.
        
        Args:
            targets: Raw targets [batch_size, output_dim]
            
        Returns:
            Scaled targets [batch_size, output_dim]
        """
        if not self._is_scaler_fitted or self.scaler is None:
            return targets
            
        # Handle tensor conversion and device
        device = targets.device
        targets_np = targets.detach().cpu().numpy()
        
        # Preserve NaN values
        nan_mask = np.isnan(targets_np)
        
        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)
            
        # Scale only non-NaN values
        scaled_np = targets_np.copy()
        valid_mask = ~nan_mask.any(axis=1) if targets_np.ndim > 1 else ~nan_mask
        if np.any(valid_mask):
            scaled_np[valid_mask] = self.scaler.transform(targets_np[valid_mask])
        
        # Convert back to tensor
        if targets.ndim == 1:
            scaled_np = scaled_np.flatten()
            
        return torch.tensor(scaled_np, dtype=targets.dtype, device=device)
    
    def unscale_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Unscale predictions to original target scale.
        
        Args:
            predictions: Scaled predictions [batch_size, output_dim]
            
        Returns:
            Unscaled predictions [batch_size, output_dim]
        """
        if not self._is_scaler_fitted or self.scaler is None:
            return predictions
            
        device = predictions.device
        predictions_np = predictions.detach().cpu().numpy()
        
        if predictions_np.ndim == 1:
            predictions_np = predictions_np.reshape(-1, 1)
            
        unscaled_np = self.scaler.inverse_transform(predictions_np)
        
        if predictions.ndim == 1:
            unscaled_np = unscaled_np.flatten()
            
        return torch.tensor(unscaled_np, dtype=predictions.dtype, device=device)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task='{self.task_def.name}', type='{self.task_def.task_type.value}')"


class RegressionTaskHead(TaskHead):
    """Task head for regression tasks."""
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute regression loss (MSE or MAE)."""
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid (non-NaN) targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            # No valid targets, return zero loss
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        
        # Compute loss based on task definition
        if self.task_def.loss_function == LossFunction.MSE:
            return F.mse_loss(valid_predictions, valid_targets)
        elif self.task_def.loss_function == LossFunction.MAE:
            return F.l1_loss(valid_predictions, valid_targets)
        else:
            raise ValueError(f"Unsupported loss function for regression: {self.task_def.loss_function}")
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute regression metrics (MSE, MAE, R²)."""
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            return {"mse": float('nan'), "mae": float('nan'), "r2": float('nan')}
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        
        # Convert to CPU for metrics computation (handle bfloat16)
        pred_np = valid_predictions.detach().cpu().float().numpy()
        target_np = valid_targets.detach().cpu().float().numpy()
        
        # Compute metrics
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        # R² score
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
        
        return {
            "mse": float(mse),
            "mae": float(mae), 
            "r2": float(r2)
        }


class BinaryClassificationTaskHead(TaskHead):
    """Task head for binary classification tasks."""
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute binary classification loss (BCE)."""
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        
        return F.binary_cross_entropy_with_logits(valid_predictions, valid_targets)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute binary classification metrics (accuracy, precision, recall, F1)."""
        
        # Flatten tensors  
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            return {"accuracy": float('nan'), "precision": float('nan'), "recall": float('nan'), "f1": float('nan')}
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        
        # Convert predictions to probabilities and binary predictions
        probs = torch.sigmoid(valid_predictions)
        binary_preds = (probs > 0.5).float()
        
        # Convert to CPU for metrics computation
        pred_np = binary_preds.detach().cpu().numpy()
        target_np = valid_targets.detach().cpu().numpy()
        
        # Compute metrics
        tp = np.sum((pred_np == 1) & (target_np == 1))
        tn = np.sum((pred_np == 0) & (target_np == 0))
        fp = np.sum((pred_np == 1) & (target_np == 0))
        fn = np.sum((pred_np == 0) & (target_np == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }


class MultiClassificationTaskHead(TaskHead):
    """Task head for multi-class classification tasks."""
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute multi-class classification loss (CE)."""
        
        # predictions: [batch_size, num_classes]
        # targets: [batch_size] (class indices)
        
        if targets.dim() > 1:
            targets = targets.flatten()
        
        # Create mask for valid targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask].long()
        
        return F.cross_entropy(valid_predictions, valid_targets)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute multi-class classification metrics (accuracy, top-k accuracy)."""
        
        if targets.dim() > 1:
            targets = targets.flatten()
        
        # Create mask for valid targets
        if mask is None:
            mask = ~torch.isnan(targets)
        else:
            mask = mask & ~torch.isnan(targets)
        
        if not mask.any():
            return {"accuracy": float('nan'), "top5_accuracy": float('nan')}
        
        # Apply mask
        valid_predictions = predictions[mask]
        valid_targets = targets[mask].long()
        
        # Convert to CPU for metrics computation (handle bfloat16)
        pred_np = valid_predictions.detach().cpu().float().numpy()
        target_np = valid_targets.detach().cpu().float().numpy()
        
        # Compute top-1 accuracy
        predicted_classes = np.argmax(pred_np, axis=1)
        accuracy = np.mean(predicted_classes == target_np)
        
        # Compute top-5 accuracy (if applicable)
        top5_accuracy = float('nan')
        if pred_np.shape[1] >= 5:
            top5_preds = np.argsort(pred_np, axis=1)[:, -5:]
            top5_accuracy = np.mean([target in top5 for target, top5 in zip(target_np, top5_preds)])
        
        return {
            "accuracy": float(accuracy),
            "top5_accuracy": float(top5_accuracy)
        }


def create_task_head(
    task_definition: TaskDefinition,
    input_dim: int,
    **kwargs
) -> TaskHead:
    """
    Factory function to create appropriate task head based on task type.
    
    Args:
        task_definition: Task definition containing type and parameters
        input_dim: Input feature dimension
        **kwargs: Additional arguments passed to task head constructor
        
    Returns:
        Appropriate TaskHead instance
        
    Raises:
        ValueError: If task type is not supported
    """
    
    task_head_classes = {
        TaskType.REGRESSION: RegressionTaskHead,
        TaskType.BINARY_CLASSIFICATION: BinaryClassificationTaskHead,
        TaskType.MULTI_CLASSIFICATION: MultiClassificationTaskHead
    }
    
    if task_definition.task_type not in task_head_classes:
        raise ValueError(f"Unsupported task type: {task_definition.task_type}")
    
    task_head_class = task_head_classes[task_definition.task_type]
    return task_head_class(task_definition, input_dim, **kwargs)


class TaskHeadCollection(nn.Module):
    """
    Collection of task heads for multi-task learning.
    
    Manages multiple TaskHead instances and provides unified interface
    for forward pass, loss computation, and metrics aggregation.
    """
    
    def __init__(self, task_heads: Dict[str, TaskHead]):
        super().__init__()
        
        if not task_heads:
            raise ValueError("At least one task head must be provided")
        
        self.task_heads = nn.ModuleDict(task_heads)
        self.task_names = list(task_heads.keys())
        
        logger.info(f"Created TaskHeadCollection with {len(self.task_heads)} task heads: {self.task_names}")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary mapping task names to predictions
        """
        predictions = {}
        for task_name, task_head in self.task_heads.items():
            predictions[task_name] = task_head(features)
        return predictions
    
    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_column_mapping: Dict[str, int],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted loss across all tasks.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Target tensor [batch_size, num_tasks]
            task_column_mapping: Maps task names to column indices in targets
            weights: Optional task weights (defaults to equal weighting)
            
        Returns:
            Tuple of (total_weighted_loss, individual_task_losses)
        """
        if weights is None:
            weights = {task_name: 1.0 for task_name in self.task_names}
        
        task_losses = {}
        total_loss = 0.0
        total_weight = 0.0
        
        for task_name, task_head in self.task_heads.items():
            if task_name in predictions:
                # Get task-specific targets
                task_column = task_column_mapping[task_name]
                task_targets = targets[:, task_column]
                
                # Compute task loss
                task_loss = task_head.compute_loss(predictions[task_name], task_targets)
                task_losses[task_name] = task_loss
                
                # Add to weighted total
                task_weight = weights.get(task_name, 1.0)
                total_loss += task_weight * task_loss
                total_weight += task_weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            # Fallback: return zero loss with gradient
            device = next(iter(predictions.values())).device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss, task_losses
    
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_column_mapping: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all tasks.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Target tensor [batch_size, num_tasks] 
            task_column_mapping: Maps task names to column indices in targets
            
        Returns:
            Dictionary mapping task names to their metrics dictionaries
        """
        all_metrics = {}
        
        for task_name, task_head in self.task_heads.items():
            if task_name in predictions:
                # Get task-specific targets
                task_column = task_column_mapping[task_name]
                task_targets = targets[:, task_column]
                
                # Compute task metrics
                task_metrics = task_head.compute_metrics(predictions[task_name], task_targets)
                all_metrics[task_name] = task_metrics
        
        return all_metrics
    
    def fit_scalers(self, targets: torch.Tensor, task_column_mapping: Dict[str, int]) -> None:
        """
        Fit scalers for all regression task heads.
        
        Args:
            targets: Training targets [num_samples, num_tasks]
            task_column_mapping: Maps task names to column indices in targets
        """
        targets_np = targets.detach().cpu().numpy()
        
        for task_name, task_head in self.task_heads.items():
            if task_head.task_def.task_type == TaskType.REGRESSION:
                task_column = task_column_mapping[task_name]
                task_targets = targets_np[:, task_column]
                task_head.fit_scaler(task_targets)
    
    def __len__(self) -> int:
        return len(self.task_heads)
    
    def __iter__(self):
        return iter(self.task_heads.items())
    
    def __contains__(self, task_name: str) -> bool:
        return task_name in self.task_heads