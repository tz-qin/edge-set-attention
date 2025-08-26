"""
Base Estimator for ESA with Clean Abstractions

This module provides the base estimator class that contains all common ESA
functionality while defining clean abstractions for task-specific implementations.

Design Principles:
- Single responsibility: base class handles only common ESA logic
- Open for extension: easy to add new task configurations
- Template method pattern: defines algorithm structure, subclasses implement specifics
- Clear separation between model architecture and task-specific logic
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import bitsandbytes as bnb
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

from esa.task_config import TaskConfiguration, TaskDefinition, TaskType
from esa.data_module import DatasetInfo
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP
from utils.norm_layers import BN, LN
from utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder

from torch_geometric.utils import to_dense_batch

logger = logging.getLogger(__name__)


def nearest_multiple_of_8(n: int) -> int:
    """Round up to nearest multiple of 8 for efficient attention computation."""
    return math.ceil(n / 8) * 8


class BaseEstimator(pl.LightningModule, ABC):
    """
    Abstract base class for all ESA estimators.
    
    Contains all common ESA functionality including:
    - Edge-Set-Attention architecture
    - Positional encodings
    - Common training utilities
    - Molecular descriptor integration
    
    Subclasses implement task-specific prediction logic.
    """
    
    def __init__(
        self,
        task_config: TaskConfiguration,
        dataset_info: DatasetInfo,
        # ESA Architecture parameters
        graph_dim: int = 256,
        hidden_dims: List[int] = None,
        num_heads: List[int] = None,
        layer_types: List[str] = None,
        apply_attention_on: str = "edge",
        # Training parameters
        lr: float = 0.001,
        batch_size: int = 32,
        optimiser_weight_decay: float = 1e-3,
        gradient_clip_val: float = 0.5,
        early_stopping_patience: int = 30,
        # Regularization parameters
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        attn_residual_dropout: float = 0.0,
        pma_residual_dropout: float = 0.0,
        # MLP parameters
        use_mlps: bool = True,
        mlp_hidden_size: int = 128,
        mlp_type: str = "standard",
        mlp_layers: int = 3,
        mlp_dropout: float = 0.0,
        use_mlp_ln: bool = False,
        # Architecture options
        norm_type: str = "LN",
        pre_or_post: str = "post",
        xformers_or_torch_attn: str = "xformers",
        use_bfloat16: bool = True,
        # Positional encoding
        posenc: Optional[str] = None,
        # Molecular descriptors
        use_molecular_descriptors: bool = False,
        # Monitoring
        monitor_loss_name: str = "val_loss",
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.task_config = task_config
        self.dataset_info = dataset_info
        
        # Architecture parameters
        self.graph_dim = graph_dim
        self.hidden_dims = hidden_dims or [256, 256, 256, 256]
        self.num_heads = num_heads or [8, 8, 8, 8]
        self.layer_types = layer_types or ["M", "S", "M", "P"]
        self.apply_attention_on = apply_attention_on
        
        # Training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.optimiser_weight_decay = optimiser_weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        
        # Regularization
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.attn_residual_dropout = attn_residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        
        # MLP parameters
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_type = mlp_type
        self.mlp_layers = mlp_layers
        self.mlp_dropout = mlp_dropout
        self.use_mlp_ln = use_mlp_ln
        
        # Architecture options
        self.norm_type = norm_type
        self.pre_or_post = pre_or_post
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.use_bfloat16 = use_bfloat16
        
        # Positional encoding
        self.posenc = posenc
        
        # Molecular descriptors
        self.use_molecular_descriptors = use_molecular_descriptors
        
        # Monitoring
        self.monitor_loss_name = monitor_loss_name
        
        # Validation
        self._validate_configuration()
        
        # Initialize components
        self._setup_positional_encoders()
        self._setup_input_projection()
        self._setup_esa_layers()
        
        # Initialize task-specific components (implemented by subclasses)
        self._setup_task_heads()
        
        # Training state tracking (minimal to save memory)
        self.test_outputs = defaultdict(list)  # Only keep test outputs
        
        logger.info(f"Initialized {self.__class__.__name__} with {task_config}")
    
    def _validate_configuration(self) -> None:
        """Validate estimator configuration."""
        # Validate dimensions match
        if len(self.hidden_dims) != len(self.num_heads):
            raise ValueError(f"hidden_dims ({len(self.hidden_dims)}) and num_heads ({len(self.num_heads)}) must have same length")
        
        if len(self.hidden_dims) != len(self.layer_types):
            raise ValueError(f"hidden_dims ({len(self.hidden_dims)}) and layer_types ({len(self.layer_types)}) must have same length")
        
        # Validate layer types
        valid_layer_types = {"M", "S", "P"}
        invalid_types = set(self.layer_types) - valid_layer_types
        if invalid_types:
            raise ValueError(f"Invalid layer types: {invalid_types}. Valid types: {valid_layer_types}")
        
        # Validate apply_attention_on
        if self.apply_attention_on not in ["node", "edge"]:
            raise ValueError(f"apply_attention_on must be 'node' or 'edge', got '{self.apply_attention_on}'")
        
        # Validate norm_type
        if self.norm_type not in ["BN", "LN"]:
            raise ValueError(f"norm_type must be 'BN' or 'LN', got '{self.norm_type}'")
    
    def _setup_positional_encoders(self) -> None:
        """Setup positional encoding components."""
        self.rwse_encoder = None
        self.lap_encoder = None
        
        if self.posenc:
            if "RWSE" in self.posenc:
                self.rwse_encoder = KernelPENodeEncoder()
                logger.info("Initialized RWSE positional encoder")
                
            if "LapPE" in self.posenc:
                self.lap_encoder = LapPENodeEncoder()
                logger.info("Initialized Laplacian positional encoder")
    
    def _setup_input_projection(self) -> None:
        """Setup input projection layers."""
        # Calculate input dimension
        base_input_dim = self.dataset_info.num_features
        
        # Add positional encoding dimensions
        if self.rwse_encoder is not None:
            base_input_dim += 24
        if self.lap_encoder is not None:
            base_input_dim += 4
        
        # Setup norm layer
        norm_fn = BN if self.norm_type == "BN" else LN
        
        if self.apply_attention_on == "node":
            # Node-based attention: project node features directly
            self.input_dim = base_input_dim
            
            if self.mlp_type in ["standard", "gated_mlp"]:
                self.node_projection = SmallMLP(
                    in_dim=self.input_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=max(1, self.mlp_layers - 1)
                )
        
        elif self.apply_attention_on == "edge":
            # Edge-based attention: concatenate source, target node features + edge features
            self.input_dim = base_input_dim * 2  # Source + target node features
            if self.dataset_info.edge_dim > 0:
                self.input_dim += self.dataset_info.edge_dim
            
            if self.mlp_type in ["standard", "gated_mlp"]:
                self.node_edge_projection = SmallMLP(
                    in_dim=self.input_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.mlp_layers
                )
        
        self.projection_norm = norm_fn(self.hidden_dims[0])
        logger.info(f"Setup input projection: {self.input_dim} -> {self.hidden_dims[0]}")
    
    def _setup_esa_layers(self) -> None:
        """Setup Edge-Set-Attention layers."""
        esa_args = {
            "num_outputs": 32,  # PMA output size
            "dim_output": self.graph_dim,
            "xformers_or_torch_attn": self.xformers_or_torch_attn,
            "dim_hidden": self.hidden_dims,
            "num_heads": self.num_heads,
            "sab_dropout": self.sab_dropout,
            "mab_dropout": self.mab_dropout,
            "pma_dropout": self.pma_dropout,
            "use_mlps": self.use_mlps,
            "mlp_hidden_size": self.mlp_hidden_size,
            "mlp_type": self.mlp_type,
            "norm_type": self.norm_type,
            "node_or_edge": self.apply_attention_on,
            "residual_dropout": self.attn_residual_dropout,
            "set_max_items": nearest_multiple_of_8(max(self.dataset_info.max_nodes, self.dataset_info.max_edges) + 1),
            "use_bfloat16": self.use_bfloat16,
            "layer_types": self.layer_types,
            "num_mlp_layers": self.mlp_layers,
            "pre_or_post": self.pre_or_post,
            "pma_residual_dropout": self.pma_residual_dropout,
            "use_mlp_ln": self.use_mlp_ln,
            "mlp_dropout": self.mlp_dropout
        }
        
        self.esa = ESA(**esa_args)
        logger.info("Initialized Edge-Set-Attention layers")
    
    @abstractmethod
    def _setup_task_heads(self) -> None:
        """Setup task-specific prediction heads. Implemented by subclasses."""
        pass
    
    def _compute_input_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Any
    ) -> torch.Tensor:
        """
        Compute input features for ESA layers.
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch object containing additional data
            
        Returns:
            Input features for ESA [num_items, input_dim]
        """
        x = x.float()
        
        # Add positional encodings
        if self.lap_encoder is not None and hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs'):
            lap_pos_enc = self.lap_encoder(batch.EigVals, batch.EigVecs)
            x = torch.cat((x, lap_pos_enc), dim=1)
            
        if self.rwse_encoder is not None and hasattr(batch, 'pestat_RWSE'):
            rwse_pos_enc = self.rwse_encoder(batch.pestat_RWSE)
            x = torch.cat((x, rwse_pos_enc), dim=1)
        
        # Project based on attention type
        if self.apply_attention_on == "edge":
            # Concatenate source and target node features
            source_features = x[edge_index[0, :], :]
            target_features = x[edge_index[1, :], :]
            edge_features = torch.cat((source_features, target_features), dim=1)
            
            # Add edge attributes if available
            if edge_attr is not None and self.dataset_info.edge_dim > 0:
                edge_features = torch.cat((edge_features, edge_attr.float()), dim=1)
            
            # Project edge features
            projected_features = self.node_edge_projection(edge_features)
            
            return projected_features
            
        else:  # apply_attention_on == "node"
            # Project node features directly
            projected_features = self.projection_norm(self.node_projection(x))
            
            return projected_features
    
    def _forward_esa(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Any
    ) -> torch.Tensor:
        """
        Forward pass through ESA layers to get graph representations.
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch_mapping: Batch assignment for nodes [num_nodes]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch object
            
        Returns:
            Graph representations [batch_size, graph_dim]
        """
        # Compute input features
        input_features = self._compute_input_features(x, edge_index, edge_attr, batch)
        
        # Determine max items and batch indices based on attention type
        if self.apply_attention_on == "edge":
            # Use edge-based batching
            max_items = batch.max_edge_global.max().item()
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
            
        else:  # apply_attention_on == "node"
            # Use node-based batching
            max_items = batch.max_node_global.max().item()
        
        # Ensure max_items is multiple of 8 for efficient attention
        max_items = nearest_multiple_of_8(max_items + 1)
        
        # Convert to dense batch format with the rounded max_items
        if self.apply_attention_on == "edge":
            dense_features, _ = to_dense_batch(
                input_features, 
                edge_batch_index, 
                fill_value=0, 
                max_num_nodes=max_items
            )
        else:  # apply_attention_on == "node"
            dense_features, _ = to_dense_batch(
                input_features,
                batch_mapping,
                fill_value=0,
                max_num_nodes=max_items
            )
        
        # Forward through ESA
        graph_representations = self.esa(
            dense_features, 
            edge_index, 
            batch_mapping, 
            num_max_items=max_items
        )
        
        return graph_representations
    
    def _get_final_features(
        self,
        graph_representations: torch.Tensor,
        batch: Any
    ) -> torch.Tensor:
        """
        Get final features for prediction by optionally adding molecular descriptors.
        
        Args:
            graph_representations: Graph representations from ESA [batch_size, graph_dim]
            batch: Batch object potentially containing molecular descriptors
            
        Returns:
            Final features for prediction [batch_size, final_dim]
        """
        if not self.use_molecular_descriptors or not hasattr(batch, 'molecular_descriptors'):
            return graph_representations
        
        # Extract and reshape molecular descriptors
        mol_desc = batch.molecular_descriptors.float()
        batch_size = graph_representations.shape[0]
        
        # Reshape from flat tensor to batch format
        mol_desc = mol_desc.view(batch_size, self.dataset_info.molecular_descriptor_dim)
        
        # Validate shapes
        assert mol_desc.shape[0] == graph_representations.shape[0], \
            f"Batch size mismatch: graph={graph_representations.shape[0]}, mol_desc={mol_desc.shape[0]}"
        
        # Concatenate graph representations with molecular descriptors
        final_features = torch.cat([graph_representations, mol_desc], dim=-1)
        
        return final_features
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through model. Implemented by subclasses.
        
        Returns either single tensor (single-task) or dict of tensors (multi-task).
        """
        pass
    
    @abstractmethod
    def _compute_loss(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        batch: Any
    ) -> torch.Tensor:
        """Compute task-specific loss. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _compute_metrics(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        batch: Any,
        step_type: str = "train"
    ) -> Dict[str, float]:
        """Compute task-specific metrics. Implemented by subclasses."""
        pass
    
    def _step(self, batch: Any, step_type: str) -> torch.Tensor:
        """
        Common training/validation/test step logic.
        
        Args:
            batch: Batch of data
            step_type: One of 'train', 'val', 'test'
            
        Returns:
            Loss tensor
        """
        # Extract batch components
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        targets = batch.y
        batch_mapping = batch.batch
        
        # Forward pass
        predictions = self.forward(x, edge_index, batch_mapping, edge_attr, batch)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets, batch)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, batch, step_type)
        
        # Log metrics (don't show in progress bar since train_loss is sufficient)
        batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.batch.max().item() + 1
        for metric_name, metric_value in metrics.items():
            self.log(f"{step_type}_{metric_name}", metric_value, prog_bar=False, batch_size=batch_size)
        
        # Skip storing outputs to save memory - only store for test if needed
        if step_type == "test":
            # Only store test outputs for final evaluation
            self.test_outputs[len(self.test_outputs)].append((predictions, targets))
        
        return loss
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss = self._step(batch, "train")
        batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.batch.max().item() + 1
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss = self._step(batch, "val")
        batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.batch.max().item() + 1
        self.log("val_loss", loss, batch_size=batch_size)
        return loss
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss = self._step(batch, "test")
        batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.batch.max().item() + 1
        self.log("test_loss", loss, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Use AdamW with bitsandbytes for efficiency
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.optimiser_weight_decay
        )
        
        # Configure learning rate scheduler - use simple val_loss for monitoring
        monitor_metric = "val_loss"  # Use consistent metric name
        mode = "max" if "MCC" in monitor_metric else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=0.5,
            patience=self.early_stopping_patience // 2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    @property
    def final_feature_dim(self) -> int:
        """Get final feature dimension for prediction heads."""
        base_dim = self.graph_dim
        if self.use_molecular_descriptors:
            base_dim += self.dataset_info.molecular_descriptor_dim
        return base_dim