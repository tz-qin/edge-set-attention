"""
Clean ESA Models with Proper Multi-Task Architecture

This module provides a clean interface that replaces the original problematic models.py
with proper separation of concerns and delegation to the new clean architecture.

This file maintains backward compatibility with existing training scripts while
using the new clean multi-task architecture under the hood.
"""

import torch
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Union
import logging

from esa.task_config import TaskConfiguration, create_task_config_from_args
from esa.data_module import DatasetInfo
from esa.estimators import create_estimator, SingleTaskEstimator, MultiTaskEstimator
from esa.base_estimator import BaseEstimator

logger = logging.getLogger(__name__)


class Estimator(pl.LightningModule):
    """
    Clean estimator interface that maintains backward compatibility.
    
    This class acts as a facade that delegates to the appropriate clean
    implementation (SingleTaskEstimator or MultiTaskEstimator) based on
    the task configuration.
    """
    
    def __init__(
        self,
        task_type: str,
        num_features: int,
        graph_dim: int,
        edge_dim: int,
        batch_size: int = 32,
        lr: float = 0.001,
        linear_output_size: int = 1,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        xformers_or_torch_attn: str = "xformers",
        hidden_dims: List[int] = None,
        num_heads: List[int] = None,
        num_sabs: int = None,  # Deprecated - use len(hidden_dims)
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        apply_attention_on: str = "edge",
        layer_types: List[str] = None,
        use_mlps: bool = False,
        set_max_items: int = 0,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        mlp_hidden_size: int = 64,
        mlp_type: str = "standard",
        attn_residual_dropout: float = 0.0,
        norm_type: str = "LN",
        triu_attn_mask: bool = False,
        output_save_dir: str = None,
        use_bfloat16: bool = True,
        is_node_task: bool = False,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        posenc: str = None,
        num_mlp_layers: int = 3,
        pre_or_post: str = "pre",
        pma_residual_dropout: float = 0,
        use_mlp_ln: bool = False,
        mlp_dropout: float = 0,
        use_molecular_descriptors: bool = False,
        molecular_descriptor_dim: int = 10,
        # Multi-task parameters (for backward compatibility)
        multi_task_target_columns: List[str] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Backward compatibility warning for deprecated multi-task parameters
        if multi_task_target_columns is not None:
            logger.warning(
                "The 'multi_task_target_columns' parameter is deprecated. "
                "Use the new clean multi-task interface via esa.estimators module for new code."
            )
        
        # Create task configuration from parameters
        self.task_config = self._create_task_config(
            task_type=task_type,
            linear_output_size=linear_output_size,
            regression_loss_fn=regression_loss_fn,
            multi_task_target_columns=multi_task_target_columns
        )
        
        # Create dataset info from parameters
        self.dataset_info = DatasetInfo(
            num_features=num_features,
            edge_dim=edge_dim,
            max_nodes=set_max_items,  # Approximation for backward compatibility
            max_edges=set_max_items,
            molecular_descriptor_dim=molecular_descriptor_dim,
            task_column_mapping=self._create_task_column_mapping()
        )
        
        # Create the actual estimator using clean architecture
        self.estimator = create_estimator(
            task_config=self.task_config,
            dataset_info=self.dataset_info,
            graph_dim=graph_dim,
            hidden_dims=hidden_dims or [256, 256, 256, 256],
            num_heads=num_heads or [8, 8, 8, 8],
            layer_types=layer_types or ["M", "S", "M", "P"],
            apply_attention_on=apply_attention_on,
            lr=lr,
            batch_size=batch_size,
            optimiser_weight_decay=optimiser_weight_decay,
            gradient_clip_val=kwargs.get('gradient_clip_val', 0.5),
            early_stopping_patience=early_stopping_patience,
            sab_dropout=sab_dropout,
            mab_dropout=mab_dropout,
            pma_dropout=pma_dropout,
            attn_residual_dropout=attn_residual_dropout,
            pma_residual_dropout=pma_residual_dropout,
            use_mlps=use_mlps,
            mlp_hidden_size=mlp_hidden_size,
            mlp_type=mlp_type,
            mlp_layers=num_mlp_layers,
            mlp_dropout=mlp_dropout,
            use_mlp_ln=use_mlp_ln,
            norm_type=norm_type,
            pre_or_post=pre_or_post,
            xformers_or_torch_attn=xformers_or_torch_attn,
            use_bfloat16=use_bfloat16,
            posenc=posenc,
            use_molecular_descriptors=use_molecular_descriptors,
            monitor_loss_name=monitor_loss_name
        )
        
        # Store additional attributes for backward compatibility
        self.task_type = task_type
        self.num_features = num_features
        self.graph_dim = graph_dim
        self.edge_dim = edge_dim
        self.batch_size = batch_size
        self.lr = lr
        self.linear_output_size = linear_output_size
        self.scaler = scaler
        self.output_save_dir = output_save_dir
        self.is_node_task = is_node_task
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        # Legacy multi-task attributes for backward compatibility
        self.multi_task_target_columns = multi_task_target_columns
        self.is_multi_task = self.task_config.is_multi_task
        self.num_tasks = len(self.task_config.tasks)
        
        logger.info(f"Created clean Estimator facade for {self.task_config}")
    
    def _create_task_config(
        self,
        task_type: str,
        linear_output_size: int,
        regression_loss_fn: str,
        multi_task_target_columns: Optional[List[str]]
    ) -> TaskConfiguration:
        """Create task configuration from legacy parameters."""
        
        # Map legacy task types
        if task_type == "multi_task_regression":
            task_type = "regression"
        
        # Create task configuration
        if multi_task_target_columns is not None and len(multi_task_target_columns) > 1:
            # Multi-task mode
            task_configs = []
            for column_name in multi_task_target_columns:
                task_name = column_name.replace("property_value::", "").replace("::", "_")
                task_configs.append({
                    "name": task_name,
                    "column_name": column_name,
                    "task_type": task_type,
                    "loss_function": regression_loss_fn
                })
            
            return TaskConfiguration.from_multi_task(task_configs)
        else:
            # Single-task mode
            target_name = multi_task_target_columns[0] if multi_task_target_columns else "target"
            return TaskConfiguration.from_single_task(
                task_name=target_name,
                column_name=target_name,
                task_type=task_type,
                loss_function=regression_loss_fn,
                num_classes=linear_output_size if task_type != "regression" else None
            )
    
    def _create_task_column_mapping(self) -> Dict[str, int]:
        """Create task column mapping for backward compatibility."""
        mapping = {}
        for i, task in enumerate(self.task_config.tasks):
            mapping[task.name] = i
        return mapping
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: torch.Tensor,
        num_max_items: int,
        batch,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass - delegate to underlying estimator."""
        return self.estimator.forward(x, edge_index, batch_mapping, edge_attr, batch)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step - delegate to underlying estimator."""
        return self.estimator.training_step(batch, batch_idx)
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Validation step - delegate to underlying estimator.""" 
        return self.estimator.validation_step(batch, batch_idx)
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step - delegate to underlying estimator."""
        return self.estimator.test_step(batch, batch_idx)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers - delegate to underlying estimator."""
        return self.estimator.configure_optimizers()
    
    # Additional methods for backward compatibility
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if hasattr(self.estimator, 'on_train_epoch_end'):
            self.estimator.on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if hasattr(self.estimator, 'on_validation_epoch_end'):
            self.estimator.on_validation_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if hasattr(self.estimator, 'on_test_epoch_end'):
            self.estimator.on_test_epoch_end()
    
    @property
    def device(self) -> torch.device:
        """Get device from underlying estimator."""
        return self.estimator.device
    
    def to(self, device) -> 'Estimator':
        """Move to device."""
        self.estimator = self.estimator.to(device)
        return super().to(device)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict from underlying estimator."""
        return self.estimator.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state dict to underlying estimator."""
        return self.estimator.load_state_dict(state_dict, strict=strict)


def create_clean_estimator(
    task_type: str,
    target_columns: Union[str, List[str]],
    dataset_info: DatasetInfo,
    **kwargs
) -> BaseEstimator:
    """
    Create a clean estimator using the new architecture.
    
    This is the recommended way to create estimators for new code.
    Use this instead of the legacy Estimator class.
    
    Args:
        task_type: Type of task ("regression", "binary_classification", "multi_classification")
        target_columns: Single target column name or list of target column names
        dataset_info: Dataset information from DataModule
        **kwargs: Additional arguments passed to estimator
        
    Returns:
        Clean estimator instance
        
    Example:
        >>> # Single-task
        >>> estimator = create_clean_estimator(
        ...     task_type="regression",
        ...     target_columns="property_value::Mic-CRO",
        ...     dataset_info=dataset_info,
        ...     graph_dim=256,
        ...     lr=0.001
        ... )
        
        >>> # Multi-task  
        >>> estimator = create_clean_estimator(
        ...     task_type="regression",
        ...     target_columns=["property_value::Mic-CRO", "property_value::Mic-RADME"],
        ...     dataset_info=dataset_info,
        ...     graph_dim=256,
        ...     lr=0.001
        ... )
    """
    # Create task configuration
    if isinstance(target_columns, str):
        # Single-task
        task_config = TaskConfiguration.from_single_task(
            task_name=target_columns,
            column_name=target_columns,
            task_type=task_type,
            loss_function=kwargs.get('regression_loss_fn', 'mse') if task_type == 'regression' else 'ce'
        )
    else:
        # Multi-task
        task_configs = []
        for column_name in target_columns:
            task_name = column_name.replace("property_value::", "").replace("::", "_")
            task_configs.append({
                "name": task_name,
                "column_name": column_name,
                "task_type": task_type,
                "loss_function": kwargs.get('regression_loss_fn', 'mse') if task_type == 'regression' else 'ce'
            })
        task_config = TaskConfiguration.from_multi_task(task_configs)
    
    # Create and return estimator
    return create_estimator(
        task_config=task_config,
        dataset_info=dataset_info,
        **kwargs
    )