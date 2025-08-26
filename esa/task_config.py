"""
Task Configuration System for ESA Multi-Task Learning

This module provides a clean, validated configuration system for managing
both single-task and multi-task learning scenarios.

Design Principles:
- Single source of truth for task configuration
- Immutable configuration objects after validation
- Clear error messages for invalid configurations
- Type safety and comprehensive validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types with clear semantics."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification" 
    MULTI_CLASSIFICATION = "multi_classification"


class LossFunction(Enum):
    """Supported loss functions mapped to task types."""
    MSE = "mse"
    MAE = "mae" 
    BCE = "bce"
    CE = "ce"


@dataclass(frozen=True)
class TaskDefinition:
    """
    Immutable definition of a single task.
    
    Contains all metadata needed to define and validate a learning task.
    """
    name: str
    column_name: str
    task_type: TaskType
    loss_function: LossFunction
    weight: float = 1.0
    num_classes: Optional[int] = None  # Required for classification tasks
    
    def __post_init__(self):
        """Validate task definition after creation."""
        self._validate()
    
    def _validate(self):
        """Comprehensive validation of task definition."""
        if not self.name or not self.name.strip():
            raise ValueError("Task name cannot be empty")
            
        if not self.column_name or not self.column_name.strip():
            raise ValueError("Column name cannot be empty")
            
        if self.weight <= 0:
            raise ValueError(f"Task weight must be positive, got {self.weight}")
            
        # Validate loss function matches task type
        self._validate_loss_function_compatibility()
        
        # Validate classification specific requirements
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTI_CLASSIFICATION]:
            if self.num_classes is None:
                raise ValueError(f"num_classes required for {self.task_type.value} tasks")
            if self.num_classes < 2:
                raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
                
        # Binary classification should have exactly 2 classes
        if self.task_type == TaskType.BINARY_CLASSIFICATION and self.num_classes != 2:
            raise ValueError(f"Binary classification requires num_classes=2, got {self.num_classes}")
    
    def _validate_loss_function_compatibility(self):
        """Validate that loss function is compatible with task type."""
        valid_combinations = {
            TaskType.REGRESSION: [LossFunction.MSE, LossFunction.MAE],
            TaskType.BINARY_CLASSIFICATION: [LossFunction.BCE],
            TaskType.MULTI_CLASSIFICATION: [LossFunction.CE]
        }
        
        if self.loss_function not in valid_combinations[self.task_type]:
            valid_losses = [lf.value for lf in valid_combinations[self.task_type]]
            raise ValueError(
                f"Loss function '{self.loss_function.value}' not compatible with "
                f"task type '{self.task_type.value}'. Valid options: {valid_losses}"
            )
    
    @property
    def output_dim(self) -> int:
        """Get the output dimension for this task."""
        if self.task_type == TaskType.REGRESSION:
            return 1
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            return 1  # Single sigmoid output
        else:  # MULTI_CLASSIFICATION
            return self.num_classes
    
    @property
    def clean_name(self) -> str:
        """Get a clean name suitable for use as Python identifier."""
        return self.name.replace("::", "_").replace("-", "_").replace(".", "_")


@dataclass(frozen=True)
class TaskConfiguration:
    """
    Immutable configuration for single or multi-task learning.
    
    Provides a unified interface for managing task configurations with
    comprehensive validation and helper methods.
    """
    tasks: List[TaskDefinition]
    global_task_type: TaskType = field(init=False)
    
    def __post_init__(self):
        """Validate configuration after creation."""
        if not self.tasks:
            raise ValueError("At least one task must be defined")
            
        # Validate all tasks have same type for multi-task learning
        task_types = {task.task_type for task in self.tasks}
        if len(task_types) > 1:
            raise ValueError(
                f"All tasks must have the same type for multi-task learning. "
                f"Found: {[tt.value for tt in task_types]}"
            )
        
        # Set global task type
        object.__setattr__(self, 'global_task_type', next(iter(task_types)))
        
        # Validate task names are unique
        task_names = [task.name for task in self.tasks]
        if len(task_names) != len(set(task_names)):
            duplicates = [name for name in task_names if task_names.count(name) > 1]
            raise ValueError(f"Duplicate task names found: {duplicates}")
        
        # Validate column names are unique
        column_names = [task.column_name for task in self.tasks]
        if len(column_names) != len(set(column_names)):
            duplicates = [name for name in column_names if column_names.count(name) > 1]
            raise ValueError(f"Duplicate column names found: {duplicates}")
    
    @property
    def is_multi_task(self) -> bool:
        """Check if this is a multi-task configuration."""
        return len(self.tasks) > 1
    
    @property
    def task_names(self) -> List[str]:
        """Get list of task names."""
        return [task.name for task in self.tasks]
    
    @property
    def column_names(self) -> List[str]:
        """Get list of target column names."""
        return [task.column_name for task in self.tasks]
    
    @property
    def task_weights(self) -> Dict[str, float]:
        """Get mapping of task names to weights."""
        return {task.name: task.weight for task in self.tasks}
    
    @property
    def total_output_dim(self) -> int:
        """Get total output dimension across all tasks."""
        return sum(task.output_dim for task in self.tasks)
    
    def get_task(self, name: str) -> TaskDefinition:
        """Get task definition by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"Task '{name}' not found. Available tasks: {self.task_names}")
    
    def get_task_by_column(self, column_name: str) -> TaskDefinition:
        """Get task definition by column name."""
        for task in self.tasks:
            if task.column_name == column_name:
                return task
        raise KeyError(f"No task found for column '{column_name}'. Available columns: {self.column_names}")
    
    @classmethod
    def from_single_task(
        cls,
        task_name: str,
        column_name: str,
        task_type: Union[str, TaskType],
        loss_function: Union[str, LossFunction],
        num_classes: Optional[int] = None,
        weight: float = 1.0
    ) -> 'TaskConfiguration':
        """
        Create configuration for single-task learning.
        
        Args:
            task_name: Human-readable name for the task
            column_name: Column name in the dataset
            task_type: Type of task (regression, binary_classification, multi_classification)
            loss_function: Loss function to use (mse, mae, bce, ce)
            num_classes: Number of classes (required for classification)
            weight: Task weight (default 1.0)
            
        Returns:
            TaskConfiguration with single task
        """
        # Convert string enums if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if isinstance(loss_function, str):
            loss_function = LossFunction(loss_function)
            
        task = TaskDefinition(
            name=task_name,
            column_name=column_name,
            task_type=task_type,
            loss_function=loss_function,
            num_classes=num_classes,
            weight=weight
        )
        
        return cls(tasks=[task])
    
    @classmethod
    def from_multi_task(
        cls,
        task_configs: List[Dict[str, Any]]
    ) -> 'TaskConfiguration':
        """
        Create configuration for multi-task learning.
        
        Args:
            task_configs: List of task configuration dictionaries.
                         Each dict should contain: name, column_name, task_type, 
                         loss_function, and optionally num_classes, weight
                         
        Returns:
            TaskConfiguration with multiple tasks
            
        Example:
            >>> configs = [
            ...     {
            ...         "name": "MIC_CRO",
            ...         "column_name": "property_value::Mic-CRO", 
            ...         "task_type": "regression",
            ...         "loss_function": "mse"
            ...     },
            ...     {
            ...         "name": "MIC_RADME",
            ...         "column_name": "property_value::Mic-RADME",
            ...         "task_type": "regression", 
            ...         "loss_function": "mae",
            ...         "weight": 2.0
            ...     }
            ... ]
            >>> config = TaskConfiguration.from_multi_task(configs)
        """
        tasks = []
        
        for i, task_config in enumerate(task_configs):
            try:
                # Validate required fields
                required_fields = ["name", "column_name", "task_type", "loss_function"]
                missing_fields = [field for field in required_fields if field not in task_config]
                if missing_fields:
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                # Convert string enums
                task_type = TaskType(task_config["task_type"])
                loss_function = LossFunction(task_config["loss_function"])
                
                # Create task definition
                task = TaskDefinition(
                    name=task_config["name"],
                    column_name=task_config["column_name"],
                    task_type=task_type,
                    loss_function=loss_function,
                    num_classes=task_config.get("num_classes"),
                    weight=task_config.get("weight", 1.0)
                )
                
                tasks.append(task)
                
            except Exception as e:
                raise ValueError(f"Error in task config {i}: {e}") from e
        
        return cls(tasks=tasks)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        task_type = "Multi-task" if self.is_multi_task else "Single-task"
        task_list = ", ".join(f"{task.name}({task.task_type.value})" for task in self.tasks)
        return f"{task_type} Configuration: [{task_list}]"
    
    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"TaskConfiguration(tasks={self.tasks})"


def create_task_config_from_args(
    target_name: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    task_type: str = "regression",
    loss_function: str = "mse",
    num_classes: Optional[int] = None,
    task_weights: Optional[List[float]] = None
) -> TaskConfiguration:
    """
    Create TaskConfiguration from command-line style arguments.
    
    This function provides backward compatibility with existing argument parsing
    while enforcing proper validation through the new configuration system.
    
    Args:
        target_name: Single target column name (for single-task)
        target_columns: List of target column names (for multi-task)
        task_type: Task type string
        loss_function: Loss function string  
        num_classes: Number of classes for classification
        task_weights: Optional list of task weights (multi-task only)
        
    Returns:
        Validated TaskConfiguration
        
    Raises:
        ValueError: If arguments are invalid or incompatible
    """
    # Validate arguments
    if target_name is not None and target_columns is not None:
        raise ValueError("Specify either target_name (single-task) or target_columns (multi-task), not both")
    
    if target_name is None and target_columns is None:
        raise ValueError("Must specify either target_name or target_columns")
    
    # Single-task case
    if target_name is not None:
        return TaskConfiguration.from_single_task(
            task_name=target_name,
            column_name=target_name,
            task_type=task_type,
            loss_function=loss_function,
            num_classes=num_classes
        )
    
    # Multi-task case
    if len(target_columns) < 2:
        raise ValueError(f"Multi-task requires at least 2 target columns, got {len(target_columns)}")
    
    # Handle task weights
    if task_weights is None:
        task_weights = [1.0] * len(target_columns)
    elif len(task_weights) != len(target_columns):
        raise ValueError(f"Number of task weights ({len(task_weights)}) must match number of target columns ({len(target_columns)})")
    
    # Create task configs
    task_configs = []
    for i, (column_name, weight) in enumerate(zip(target_columns, task_weights)):
        # Generate a clean task name from column name
        task_name = column_name.replace("property_value::", "").replace("::", "_")
        
        task_configs.append({
            "name": task_name,
            "column_name": column_name,
            "task_type": task_type,
            "loss_function": loss_function,
            "num_classes": num_classes,
            "weight": weight
        })
    
    return TaskConfiguration.from_multi_task(task_configs)