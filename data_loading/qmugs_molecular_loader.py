"""
QMugs Dataset Loader for ESA Transfer Learning

This module provides functionality to load pre-featurized QMugs dataset 
for ESA transfer learning experiments.

The QMugs dataset should be provided as a parquet file that already contains
the same featurization format as your basic_gnn model (gnn2d_atom_features, 
gnn2d_bond_features, molecular descriptors, and QM target properties).
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Dict, Any
import os
import hashlib

from data_loading.transforms import AddMaxEdge, AddMaxNode, AddMaxEdgeGlobal, AddMaxNodeGlobal, FormatSingleLabel
import torch_geometric.transforms as T


class QMugsDataset(InMemoryDataset):
    """
    QMugs dataset loader for pre-featurized parquet files.
    
    Loads QMugs data that has already been featurized to match your 
    basic_gnn format (gnn2d features + molecular descriptors + QM targets).
    """
    
    def __init__(
        self,
        qmugs_parquet_path: str,
        target_properties: List[str],
        transform=None,
        pre_transform=None, 
        pre_filter=None,
    ):
        """
        Initialize QMugs dataset from pre-featurized parquet.
        
        Args:
            qmugs_parquet_path: Path to pre-featurized QMugs parquet file
            target_properties: List of QM properties to extract (e.g., ['homo', 'lumo', 'gap'])
            transform: PyG transform to apply on-the-fly
            pre_transform: PyG transform to apply during processing
            pre_filter: PyG filter to apply during processing
        """
        self.qmugs_parquet_path = qmugs_parquet_path
        self.target_properties = target_properties
        
        # Create unique cache directory
        qmugs_name = os.path.basename(qmugs_parquet_path)
        path_hash = hashlib.md5(qmugs_parquet_path.encode()).hexdigest()[:8]
        props_str = "_".join(sorted(target_properties))
        
        cache_dir = f"./cache_qmugs_{qmugs_name}_{props_str}_{path_hash}"
        
        super().__init__(cache_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return [self.qmugs_parquet_path.split('/')[-1]]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # No download needed - data is already local
        pass
    
    def process(self):
        """Process pre-featurized QMugs parquet into PyG format."""
        print(f"Loading pre-featurized QMugs data from {self.qmugs_parquet_path}")
        
        df = pd.read_parquet(self.qmugs_parquet_path)
        print(f"QMugs dataset contains {len(df)} molecules")
        print(f"Available columns: {list(df.columns)}")
        
        # Filter molecules that have at least one valid target property
        valid_molecules = []
        for prop in self.target_properties:
            if prop in df.columns:
                valid_molecules.append(~df[prop].isna())
            else:
                print(f"Warning: Property '{prop}' not found in dataset")
        
        if valid_molecules:
            valid_mask = np.logical_or.reduce(valid_molecules)
            df = df[valid_mask]
            print(f"After filtering for valid targets: {len(df)} molecules")
        
        data_list = []
        failed_conversions = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting QMugs molecules"):
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
        Convert a pre-featurized row to PyG Data object.
        
        Expects the row to contain the same features as your basic_gnn format:
        - gnn2d_atom_features, gnn2d_bond_features, gnn2d_atom_numbers, gnn2d_bond_idxs
        - Molecular descriptors: x_log_p, aromatic_ring_count, etc.
        - QM targets: homo, lumo, gap, etc.
        """
        try:
            # Extract GNN2D features (same as FFPM molecular loader)
            atom_features = np.array(row['gnn2d_atom_features'])
            atom_numbers = np.array(row['gnn2d_atom_numbers'])
            bond_features = np.array(row['gnn2d_bond_features'])
            bond_idxs = np.array(row['gnn2d_bond_idxs'])
            
            # Validate data shapes
            num_atoms = len(atom_features)
            num_bonds = len(bond_features)
            
            if num_atoms == 0 or num_bonds == 0:
                return None
            
            # Convert atom features to tensor
            atom_features_list = [np.array(feat, dtype=np.float32) for feat in atom_features]
            x = torch.tensor(np.stack(atom_features_list), dtype=torch.float)
            
            # Convert bond indices to edge_index
            edge_pairs = []
            for bond_idx in bond_idxs:
                if isinstance(bond_idx, np.ndarray) and len(bond_idx) == 2:
                    edge_pairs.append(bond_idx)
                else:
                    return None
            
            if len(edge_pairs) == 0:
                return None
                
            edge_index = torch.tensor(np.array(edge_pairs).T, dtype=torch.long)
            
            # Convert bond features to edge attributes
            bond_features_list = [np.array(feat, dtype=np.float32) for feat in bond_features]
            edge_attr = torch.tensor(np.stack(bond_features_list), dtype=torch.float)
            
            # Ensure edge_index and edge_attr have compatible shapes
            if edge_index.shape[1] != edge_attr.shape[0]:
                return None
            
            # Make graph undirected (ESA expects undirected graphs)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            
            # Extract QM target properties
            qm_targets = {}
            for prop in self.target_properties:
                if prop in row.index and pd.notna(row[prop]):
                    qm_targets[prop] = float(row[prop])
            
            if not qm_targets:
                return None  # Skip molecules with no valid targets
            
            # Extract molecular descriptors (same as your basic_gnn config)
            molecular_descriptor_cols = [
                'x_log_p', 'aromatic_ring_count', 'molecular_weight',
                'num_acceptors', 'num_donors', 'rotatable_bonds', 
                'tpsa', 'log_d', 'most_acidic_pka', 'most_basic_pka'
            ]
            
            # Create PyG Data object matching your basic_gnn format
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_atoms,
                
                # Store QM targets for multi-task learning
                qm_targets=qm_targets,
                
                # Store SMILES if available
                smiles=row.get('canonical_smiles', row.get('smiles', '')),
            )
            
            # Add molecular descriptors as individual attributes (same as basic_gnn)
            for desc_col in molecular_descriptor_cols:
                if desc_col in row.index and pd.notna(row[desc_col]):
                    setattr(data, desc_col, torch.tensor([float(row[desc_col])], dtype=torch.float))
                else:
                    # Set default values for missing descriptors
                    default_val = 15.0 if 'pka' in desc_col else 0.0
                    setattr(data, desc_col, torch.tensor([default_val], dtype=torch.float))
            
            return data
            
        except Exception as e:
            print(f"Error converting row: {e}")
            return None


def load_qmugs_for_multitask_pretraining(
    qmugs_parquet_path: str,
    target_properties: List[str] = ['homo', 'lumo', 'gap', 'dipole_moment', 'total_energy'],
    **kwargs
):
    """
    Load pre-featurized QMugs dataset for multi-task QM pre-training.
    
    Args:
        qmugs_parquet_path: Path to pre-featurized QMugs parquet file
        target_properties: List of QM properties to train on
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        QMugs dataset formatted for ESA multi-task training
    """
    
    # Set up transforms for ESA compatibility
    transforms = [AddMaxEdge(), AddMaxNode()]
    
    dataset = QMugsDataset(
        qmugs_parquet_path=qmugs_parquet_path,
        target_properties=target_properties,
        pre_transform=T.Compose(transforms),
    )
    
    print(f"\nQMugs dataset loaded: {len(dataset)} molecules")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Node feature dim: {sample.x.shape[1] if hasattr(sample, 'x') else 'N/A'}")
        print(f"Edge feature dim: {sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 'N/A'}")
        print(f"Available QM targets: {list(sample.qm_targets.keys()) if hasattr(sample, 'qm_targets') else 'N/A'}")
    
    # Calculate global statistics for consistent padding with your basic_gnn
    max_nodes = max([data.num_nodes for data in dataset])
    max_edges = max([data.edge_index.shape[1] for data in dataset])
    
    print(f"Max nodes per graph: {max_nodes}")
    print(f"Max edges per graph: {max_edges}")
    
    # Apply global transforms
    global_transforms = T.Compose([
        AddMaxEdgeGlobal(max_edges),
        AddMaxNodeGlobal(max_nodes)
    ])
    
    print("Applying global transforms...")
    processed_data = [global_transforms(data) for data in tqdm(dataset, desc="Processing")]
    
    from data_loading.data_loading import CustomPyGDataset
    final_dataset = CustomPyGDataset(processed_data)
    
    print(f"Final QMugs dataset: {len(final_dataset)} molecules ready for ESA pre-training")
    
    return final_dataset