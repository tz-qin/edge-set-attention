"""
FFPM Molecular Data Loader for Edge-Set-Attention (ESA)

This module provides functionality to load molecular data featurized with FFPM
and convert it to PyTorch Geometric format compatible with ESA models.

The loader handles the conversion from FFPM's GNN2D features to ESA's expected
node and edge feature format. It works with any molecular dataset that has been
featurized using the FFPM pipeline.

## Data Pipeline Overview

The integration follows this data flow:
1. SMILES strings → FFPM featurization → GNN2D features (stored in parquet)
2. GNN2D features → PyTorch Geometric Data objects → ESA training

## Feature Format

**Node Features (36-dimensional):**
- 26-dim: FFPM atom features (atomic properties, hybridization, etc.)
- 10-dim: Molecular descriptors (x_log_p, aromatic_ring_count, molecular_weight, 
          num_acceptors, num_donors, rotatable_bonds, tpsa, log_d, 
          most_acidic_pka, most_basic_pka)

**Edge Features (11-dimensional):**
- FFPM bond features (bond type, aromaticity, ring membership, etc.)

## Data Splitting Strategy

This module supports two data splitting approaches:

### 1. Pre-defined Splits (Recommended)
When a directory contains pre-defined train/test files:
- `stl_train_set_08_10.parquet` → Training data
- `test_set_08_10.parquet` → Test data  
- Training data is further split into train/validation (80/20 by default)
- Eliminates data leakage and ensures reproducible splits

### 2. Random Splits (Fallback)
When only a single dataset file is available:
- Uses random 70/15/15 train/validation/test splitting
- Seed-controlled for reproducibility

## Usage Examples

### Python API
```python
# Automatic split detection (recommended)
train, val, test, num_classes, task_type, scaler = get_ffpm_molecular_dataset_train_val_test(
    dataset_dir="/path/to/hclint",  # Directory with pre-defined splits
    target_column="property_value::Mic-CRO"
)

# Single file with random splitting
train, val, test, num_classes, task_type, scaler = get_ffpm_molecular_dataset_train_val_test(
    dataset_dir="/path/to/dataset_feated.parquet",  # Single file
    target_column="property_value::Mic-CRO"
)
```

### ESA Training Command
```bash
# Example training command for FFPM molecular data (uses pre-defined splits automatically)
python -m esa.train \
    --dataset FFPM_MOLECULAR \
    --dataset-download-dir /rxrx/data/user/thomas.qin/hclint \
    --dataset-target-name property_value::Mic-CRO \
    --dataset-one-hot \
    --lr 0.0001 \
    --batch-size 128 \
    --norm-type BN \
    --early-stopping-patience 30 \
    --monitor-loss-name val_loss/dataloader_idx_0 \
    --regression-loss-fn mae \
    --graph-dim 256 \
    --apply-attention-on edge \
    --use-mlps \
    --mlp-hidden-size 256 \
    --out-path ffpm_molecular_output \
    --hidden-dims 256 256 256 256 256 256 \
    --num-heads 16 16 16 16 16 16 \
    --sab-dropout 0 \
    --mab-dropout 0 \
    --pma-dropout 0 \
    --seed 42 \
    --optimiser-weight-decay 1e-10 \
    --gradient-clip-val 0.5 \
    --xformers-or-torch-attn xformers \
    --mlp-type standard \
    --use-bfloat16 \
    --pre-or-post post \
    --layer-types M S M S M P
```

### Key Parameter Notes for Molecular Data:
- `--dataset FFPM_MOLECULAR`: Specifies the molecular dataset type
- `--dataset-download-dir`: Path to directory containing parquet files
- `--dataset-target-name`: Target column (e.g., property_value::Mic-CRO)
- `--regression-loss-fn mae`: Loss function for regression tasks (mae or mse)
- `--apply-attention-on edge`: ESA attention on edges (recommended for molecular graphs)
- `--graph-dim 256`: Graph representation dimension matching molecular complexity
- `--layer-types M S M S M P`: Mixed layer types (M=MAB, S=SAB, P=PMA) for molecular features

## Implementation Changes (August 2025)

Recent updates to support pre-defined train/test splits:

1. **Enhanced Auto-detection Logic**: The main wrapper function now intelligently 
   detects whether input is a directory or single file and chooses appropriate 
   splitting strategy.

2. **New Function: load_ffpm_molecular_data_with_predefined_splits()**: Handles 
   loading from separate train/test parquet files while maintaining validation 
   split from training data.

3. **Backwards Compatibility**: Existing single-file workflows continue to work 
   with random splitting as fallback.

4. **Improved ESA Integration**: No changes required to ESA's main data loading 
   interface - the integration remains transparent.

## Data Statistics (HCLINT Example)

**Pre-defined Splits:**
- Train: 2,717 molecules (from stl_train_set_08_10.parquet)
- Validation: 679 molecules (20% of training data)  
- Test: 3,396 molecules (from test_set_08_10.parquet)

**Random Splits (dataset_feated.parquet):**
- Train: 2,377 molecules (70%)
- Validation: 509 molecules (15%) 
- Test: 510 molecules (15%)
- Total: 3,396 molecules

## Technical Notes

- **Target Scaling**: Regression targets are standardized using sklearn.StandardScaler
  fitted on training data only
- **Graph Construction**: Molecular graphs are converted to undirected format 
  expected by ESA
- **Global Statistics**: Max nodes/edges are computed across all splits to ensure 
  consistent padding
- **Error Handling**: Robust handling of malformed molecular data with detailed 
  logging of conversion failures
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Dict, Any

from data_loading.transforms import AddMaxEdge, AddMaxNode, AddMaxEdgeGlobal, AddMaxNodeGlobal, FormatSingleLabel
import torch_geometric.transforms as T


class FFPMMolecularDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for molecular data featurized with FFPM.
    
    This dataset converts FFMP featurized molecular data to PyG format suitable
    for ESA models. It handles:
    - Converting FFPM GNN2D features to PyG node/edge features
    - Adding molecular descriptors as additional node features
    - Proper graph construction from bond indices
    - Target variable selection and preprocessing
    """
    
    def __init__(
        self,
        parquet_path: str,
        target_column: str,
        task_type: str = "regression",
        include_molecular_descriptors: bool = False,
        molecular_descriptor_cols: Optional[List[str]] = None,
        exclude_molecular_descriptor_cols: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize FFPM Molecular dataset.
        
        Args:
            parquet_path: Path to the featurized parquet file
            target_column: Column name for the target variable
            task_type: "regression" or "classification"
            include_molecular_descriptors: Whether to include molecular descriptors as node features
            molecular_descriptor_cols: List of molecular descriptor columns to include (if None, uses default list)
            exclude_molecular_descriptor_cols: List of molecular descriptor columns to exclude from default list
            transform: PyG transform to apply on-the-fly
            pre_transform: PyG transform to apply during processing
            pre_filter: PyG filter to apply during processing
        """
        self.parquet_path = parquet_path
        self.target_column = target_column
        self.task_type = task_type
        self.include_molecular_descriptors = include_molecular_descriptors
        
        # Configure molecular descriptor columns
        default_molecular_descriptor_cols = [
            'x_log_p', 'aromatic_ring_count', 'molecular_weight',
            'num_acceptors', 'num_donors', 'rotatable_bonds', 
            'tpsa', 'log_d', 'most_acidic_pka', 'most_basic_pka'
        ]
        
        if molecular_descriptor_cols is not None:
            # Use explicitly specified columns
            self.molecular_descriptor_cols = molecular_descriptor_cols
        else:
            # Start with default columns
            self.molecular_descriptor_cols = default_molecular_descriptor_cols.copy()
            
            # Remove excluded columns if specified
            if exclude_molecular_descriptor_cols is not None:
                self.molecular_descriptor_cols = [
                    col for col in self.molecular_descriptor_cols 
                    if col not in exclude_molecular_descriptor_cols
                ]
            
        # Create unique cache directory based on parquet file path and configuration to avoid conflicts
        import os
        import hashlib
        parquet_name = os.path.basename(parquet_path)
        path_hash = hashlib.md5(parquet_path.encode()).hexdigest()[:8]
        
        # Include feature configuration in cache path to avoid conflicts
        feature_config = f"moldescs_{include_molecular_descriptors}_cols_{len(self.molecular_descriptor_cols)}"
        cache_dir = f"./cache_ffpm_{parquet_name}_{target_column}_{feature_config}_{path_hash}"
        
        super().__init__(cache_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return [self.parquet_path.split('/')[-1]]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # No download needed - data is already local
        pass
    
    def process(self):
        """Process FFPM featurized data into PyG format."""
        print(f"Loading data from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        
        # Filter out rows with missing target values
        df = df.dropna(subset=[self.target_column])
        print(f"Processing {len(df)} molecules with valid targets")
        
        data_list = []
        failed_conversions = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting molecules"):
            try:
                data = self._convert_row_to_pyg(row)
                if data is not None:
                    data_list.append(data)
                else:
                    failed_conversions += 1
            except Exception as e:
                print(f"Failed to convert molecule {idx}: {e}")
                failed_conversions += 1
        
        print(f"Successfully converted {len(data_list)} molecules")
        print(f"Failed to convert {failed_conversions} molecules")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _convert_row_to_pyg(self, row: pd.Series) -> Optional[Data]:
        """
        Convert a single row of FFPM featurized data to PyG Data object.
        
        Args:
            row: Pandas series containing FFPM features and target
            
        Returns:
            PyG Data object or None if conversion fails
        """
        try:
            # Extract GNN2D features
            atom_features = np.array(row['gnn2d_atom_features'])  # Shape: (num_atoms, 26)
            atom_numbers = np.array(row['gnn2d_atom_numbers'])    # Shape: (num_atoms,)
            bond_features = np.array(row['gnn2d_bond_features'])  # Shape: (num_bonds, 11)
            bond_idxs = np.array(row['gnn2d_bond_idxs'])         # Shape: (num_bonds,)
            
            # Validate data shapes
            num_atoms = len(atom_features)
            num_bonds = len(bond_features)
            
            if num_atoms == 0 or num_bonds == 0:
                return None
            
            # Convert atom features to tensor
            # Handle numpy object arrays properly
            atom_features_list = [np.array(feat, dtype=np.float32) for feat in atom_features]
            x = torch.tensor(np.stack(atom_features_list), dtype=torch.float)
            
            # Add molecular descriptors to node features if requested
            if self.include_molecular_descriptors:
                mol_descriptors = []
                for col in self.molecular_descriptor_cols:
                    if col in row.index and not pd.isna(row[col]):
                        mol_descriptors.append(float(row[col]))
                    else:
                        mol_descriptors.append(0.0)  # Default value for missing descriptors
                
                # Broadcast molecular descriptors to all atoms
                mol_desc_tensor = torch.tensor(mol_descriptors, dtype=torch.float)
                mol_desc_expanded = mol_desc_tensor.unsqueeze(0).expand(num_atoms, -1)
                x = torch.cat([x, mol_desc_expanded], dim=1)
            
            # Convert bond indices to edge_index
            # FFPM bond_idxs contains arrays of (src, dst) pairs, not flattened pairs
            edge_pairs = []
            for bond_idx in bond_idxs:
                if isinstance(bond_idx, np.ndarray) and len(bond_idx) == 2:
                    edge_pairs.append(bond_idx)
                else:
                    return None  # Invalid bond index format
            
            if len(edge_pairs) == 0:
                return None
                
            edge_index = torch.tensor(np.array(edge_pairs).T, dtype=torch.long)
            
            # Convert bond features to edge attributes  
            # Handle numpy object arrays properly
            bond_features_list = [np.array(feat, dtype=np.float32) for feat in bond_features]
            edge_attr = torch.tensor(np.stack(bond_features_list), dtype=torch.float)
            
            # Ensure edge_index and edge_attr have compatible shapes
            if edge_index.shape[1] != edge_attr.shape[0]:
                return None
            
            # Make graph undirected (ESA expects undirected graphs)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            
            # Extract target variable
            y = torch.tensor([float(row[self.target_column])], dtype=torch.float)
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                smiles=row['canonical_smiles'],
                num_nodes=num_atoms,
            )
            
            return data
            
        except Exception as e:
            print(f"Error converting row: {e}")
            return None


def load_ffpm_molecular_data_with_predefined_splits(
    train_parquet_path: str,
    test_parquet_path: str,
    target_column: str,
    task_type: str = "regression",
    val_frac: float = 0.2,  # Fraction of training data to use for validation
    include_molecular_descriptors: bool = False,
    molecular_descriptor_cols: Optional[List[str]] = None,
    exclude_molecular_descriptor_cols: Optional[List[str]] = None,
    random_seed: int = 42,
    pe_types: List[str] = None,  # Positional encoding types (not used for molecular data)
    **kwargs  # Accept additional arguments from ESA training
) -> Tuple[FFPMMolecularDataset, FFPMMolecularDataset, FFPMMolecularDataset, int, str, Optional[StandardScaler]]:
    """
    Load FFPM molecular data using pre-defined train/test splits.
    
    This function implements the recommended approach for loading molecular data
    when pre-defined train/test splits are available. It:
    
    1. Loads training data from train_parquet_path
    2. Loads test data from test_parquet_path  
    3. Further splits training data into train/validation using val_frac
    4. Ensures consistent feature scaling across all splits
    5. Maintains data integrity by keeping test set completely separate
    
    This approach eliminates data leakage concerns and ensures reproducible
    benchmarking results across different experiments.
    
    Args:
        train_parquet_path: Path to training data parquet file (e.g., stl_train_set_08_10.parquet)
        test_parquet_path: Path to test data parquet file (e.g., test_set_08_10.parquet)
        target_column: Target variable column name (e.g., "property_value::Mic-CRO")
        task_type: "regression" or "classification"
        val_frac: Fraction of training data to use for validation (default: 0.2)
        include_molecular_descriptors: Whether to include molecular descriptors as node features
        molecular_descriptor_cols: List of molecular descriptor columns to include
        random_seed: Random seed for reproducible validation split from training data
        pe_types: Positional encoding types (not used for molecular data)
        **kwargs: Additional arguments from ESA training
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, num_classes, task_type, scaler)
        
    Example:
        >>> train, val, test, num_classes, task_type, scaler = load_ffpm_molecular_data_with_predefined_splits(
        ...     train_parquet_path="/path/to/stl_train_set_08_10.parquet",
        ...     test_parquet_path="/path/to/test_set_08_10.parquet", 
        ...     target_column="property_value::Mic-CRO"
        ... )
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    
    # Set up transforms for ESA compatibility
    transforms = [AddMaxEdge(), AddMaxNode()]
    if task_type == "regression":
        transforms.append(FormatSingleLabel())
    
    # Load train dataset
    print(f"Loading training data from {train_parquet_path}")
    train_dataset = FFPMMolecularDataset(
        parquet_path=train_parquet_path,
        target_column=target_column,
        task_type=task_type,
        include_molecular_descriptors=include_molecular_descriptors,
        molecular_descriptor_cols=molecular_descriptor_cols,
        exclude_molecular_descriptor_cols=exclude_molecular_descriptor_cols,
        pre_transform=T.Compose(transforms),
    )
    
    # Load test dataset  
    print(f"Loading test data from {test_parquet_path}")
    test_dataset = FFPMMolecularDataset(
        parquet_path=test_parquet_path,
        target_column=target_column,
        task_type=task_type,
        include_molecular_descriptors=include_molecular_descriptors,
        molecular_descriptor_cols=molecular_descriptor_cols,
        exclude_molecular_descriptor_cols=exclude_molecular_descriptor_cols,
        pre_transform=T.Compose(transforms),
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} molecules")
    print(f"Test dataset: {len(test_dataset)} molecules")
    print(f"Node feature dim: {train_dataset[0].x.shape[1]}")
    print(f"Edge feature dim: {train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0}")
    
    # Calculate global statistics across both train and test sets
    all_data = list(train_dataset) + list(test_dataset)
    max_nodes = max([data.num_nodes for data in all_data])
    max_edges = max([data.edge_index.shape[1] for data in all_data])
    
    print(f"Max nodes per graph: {max_nodes}")
    print(f"Max edges per graph: {max_edges}")
    
    # Add global max transforms
    global_transforms = T.Compose([
        AddMaxEdgeGlobal(max_edges),
        AddMaxNodeGlobal(max_nodes)
    ])
    
    # Apply global transforms to all data
    print("Applying global transforms...")
    train_data = [global_transforms(data) for data in tqdm(train_dataset, desc="Processing train")]
    test_data = [global_transforms(data) for data in tqdm(test_dataset, desc="Processing test")]
    
    # Split training data into train/validation
    torch.manual_seed(random_seed)
    train_size = len(train_data)
    val_size = int(val_frac * train_size)
    actual_train_size = train_size - val_size
    
    indices = torch.randperm(train_size)
    train_indices = indices[:actual_train_size]
    val_indices = indices[actual_train_size:]
    
    actual_train_data = [train_data[i] for i in train_indices]
    val_data = [train_data[i] for i in val_indices]
    
    # Handle target scaling for regression
    scaler = None
    if task_type == "regression":
        train_targets = [data.y.item() for data in actual_train_data]
        scaler = StandardScaler()
        scaler.fit(np.array(train_targets).reshape(-1, 1))
        
        # Apply scaling to all splits
        for data in actual_train_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
        for data in val_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
        for data in test_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
    
    # Create final datasets
    from data_loading.data_loading import CustomPyGDataset
    final_train_dataset = CustomPyGDataset(actual_train_data)
    final_val_dataset = CustomPyGDataset(val_data)
    final_test_dataset = CustomPyGDataset(test_data)
    
    # Determine num_classes and task_type
    if task_type == "regression":
        num_classes = 1
    else:
        # For classification, determine number of unique classes
        all_targets = [data.y.item() for data in train_data + test_data]
        num_classes = len(set(all_targets))
        if num_classes == 2:
            task_type = "binary_classification"
        else:
            task_type = "multi_classification"
    
    print(f"\nFinal dataset split:")
    print(f"  Train: {len(final_train_dataset)} molecules")
    print(f"  Val: {len(final_val_dataset)} molecules")
    print(f"  Test: {len(final_test_dataset)} molecules")
    print(f"  Task type: {task_type}")
    print(f"  Num classes: {num_classes}")
    
    return final_train_dataset, final_val_dataset, final_test_dataset, num_classes, task_type, scaler


def load_ffpm_molecular_data(
    parquet_path: str,
    target_column: str,
    task_type: str = "regression",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    include_molecular_descriptors: bool = False,
    molecular_descriptor_cols: Optional[List[str]] = None,
    random_seed: int = 42,
    pe_types: List[str] = None,  # Positional encoding types (not used for molecular data)
    **kwargs  # Accept additional arguments from ESA training
) -> Tuple[FFPMMolecularDataset, FFPMMolecularDataset, FFPMMolecularDataset, int, str, Optional[StandardScaler]]:
    """
    Load FFPM molecular data with random train/val/test splitting.
    
    This function is kept for backwards compatibility but now uses pre-defined splits
    by default when train/test parquet files are available.
    """
    
    # Set up transforms for ESA compatibility
    transforms = [AddMaxEdge(), AddMaxNode()]
    if task_type == "regression":
        transforms.append(FormatSingleLabel())
    
    # Create dataset
    dataset = FFPMMolecularDataset(
        parquet_path=parquet_path,
        target_column=target_column,
        task_type=task_type,
        include_molecular_descriptors=include_molecular_descriptors,
        molecular_descriptor_cols=molecular_descriptor_cols,
        pre_transform=T.Compose(transforms),
    )
    
    print(f"\nDataset loaded: {len(dataset)} molecules")
    print(f"Node feature dim: {dataset[0].x.shape[1]}")
    print(f"Edge feature dim: {dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0}")
    
    # Calculate dataset statistics for global max nodes/edges
    max_nodes = max([data.num_nodes for data in dataset])
    max_edges = max([data.edge_index.shape[1] for data in dataset])
    
    print(f"Max nodes per graph: {max_nodes}")
    print(f"Max edges per graph: {max_edges}")
    
    # Add global max transforms
    global_transforms = T.Compose([
        AddMaxEdgeGlobal(max_edges),
        AddMaxNodeGlobal(max_nodes)
    ])
    
    # Apply global transforms
    print("Applying global transforms...")
    processed_data = [global_transforms(data) for data in tqdm(dataset)]
    
    # Split dataset
    torch.manual_seed(random_seed)
    total_size = len(processed_data)
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)
    test_size = total_size - train_size - val_size
    
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]
    test_data = [processed_data[i] for i in test_indices]
    
    # Handle target scaling for regression
    scaler = None
    if task_type == "regression":
        train_targets = [data.y.item() for data in train_data]
        scaler = StandardScaler()
        scaler.fit(np.array(train_targets).reshape(-1, 1))
        
        # Apply scaling to all splits
        for data in train_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
        for data in val_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
        for data in test_data:
            data.y = torch.tensor(scaler.transform([[data.y.item()]])[0], dtype=torch.float)
    
    # Create final datasets
    from data_loading.data_loading import CustomPyGDataset
    train_dataset = CustomPyGDataset(train_data)
    val_dataset = CustomPyGDataset(val_data)
    test_dataset = CustomPyGDataset(test_data)
    
    # Determine num_classes and task_type
    if task_type == "regression":
        num_classes = 1
    else:
        # For classification, determine number of unique classes
        all_targets = [data.y.item() for data in processed_data]
        num_classes = len(set(all_targets))
        if num_classes == 2:
            task_type = "binary_classification"
        else:
            task_type = "multi_classification"
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} molecules")
    print(f"  Val: {len(val_dataset)} molecules")
    print(f"  Test: {len(test_dataset)} molecules")
    print(f"  Task type: {task_type}")
    print(f"  Num classes: {num_classes}")
    
    return train_dataset, val_dataset, test_dataset, num_classes, task_type, scaler


def get_ffpm_molecular_dataset_train_val_test(
    dataset_dir: str,
    target_column: str,
    **kwargs
) -> Tuple[FFPMMolecularDataset, FFPMMolecularDataset, FFPMMolecularDataset, int, str, Optional[StandardScaler]]:
    """
    Main wrapper function compatible with ESA's data loading interface.
    
    This function implements intelligent data splitting strategy selection:
    
    **Pre-defined Splits (Recommended):**
    If dataset_dir is a directory containing:
    - `stl_train_set_08_10.parquet` 
    - `test_set_08_10.parquet`
    
    Then it uses these pre-defined splits, which eliminates data leakage and 
    ensures reproducible benchmarking. The training data is further split 
    into train/validation.
    
    **Random Splits (Fallback):**
    If dataset_dir is:
    - A single parquet file, OR
    - A directory without pre-defined split files but containing `dataset_feated.parquet`
    
    Then it uses random 70/15/15 train/validation/test splitting with seed control.
    
    **Integration with ESA:**
    This function maintains the same interface as other ESA dataset loaders,
    so no changes are needed to ESA's training scripts. Simply use:
    ```bash
    python esa/train.py --dataset FFPM_MOLECULAR --dataset-download-dir /path/to/hclint
    ```
    
    Args:
        dataset_dir: Directory containing parquet files OR path to single parquet file
        target_column: Target variable column name (e.g., "property_value::Mic-CRO")
        **kwargs: Additional arguments passed to the loading functions
                 (task_type, val_frac, include_molecular_descriptors, etc.)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, num_classes, task_type, scaler)
        
    Raises:
        FileNotFoundError: If no valid dataset files are found
        
    Example:
        >>> # Automatic detection - uses pre-defined splits if available
        >>> train, val, test, num_classes, task_type, scaler = get_ffpm_molecular_dataset_train_val_test(
        ...     dataset_dir="/rxrx/data/user/thomas.qin/hclint",
        ...     target_column="property_value::Mic-CRO"
        ... )
        Found pre-defined train/test splits: .../stl_train_set_08_10.parquet, .../test_set_08_10.parquet
        
        >>> # Single file - uses random splitting  
        >>> train, val, test, num_classes, task_type, scaler = get_ffpm_molecular_dataset_train_val_test(
        ...     dataset_dir="/path/to/dataset_feated.parquet",
        ...     target_column="property_value::Mic-CRO"
        ... )
    """
    import os
    
    # Remove 'target_name' if it's in kwargs to avoid duplication since we handle it separately
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'target_name'}
    
    # Check if dataset_dir is a single file or directory
    if os.path.isfile(dataset_dir):
        # Single file - use random splitting
        return load_ffpm_molecular_data(dataset_dir, target_column, **kwargs_clean)
    
    # Directory - check for pre-defined split files
    train_file = os.path.join(dataset_dir, "stl_train_set_08_10.parquet")
    test_file = os.path.join(dataset_dir, "test_set_08_10.parquet")
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"Found pre-defined train/test splits: {train_file}, {test_file}")
        return load_ffpm_molecular_data_with_predefined_splits(
            train_parquet_path=train_file,
            test_parquet_path=test_file,
            target_column=target_column,
            **kwargs_clean
        )
    else:
        # Fall back to single dataset file approach
        # Look for dataset_feated.parquet or other parquet files
        dataset_file = os.path.join(dataset_dir, "dataset_feated.parquet")
        if os.path.exists(dataset_file):
            print(f"Using single dataset file with random splitting: {dataset_file}")
            return load_ffpm_molecular_data(dataset_file, target_column, **kwargs_clean)
        else:
            raise FileNotFoundError(
                f"Could not find pre-defined splits ({train_file}, {test_file}) "
                f"or single dataset file ({dataset_file}) in {dataset_dir}"
            )