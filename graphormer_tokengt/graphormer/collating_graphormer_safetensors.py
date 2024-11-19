import numpy as np
import torch

from typing import Any, Dict, List
from transformers.utils import is_cython_available, requires_backends
from transformers.utils.import_utils import is_cython_available

if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer  # noqa E402


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


# TUDataset molecular datasets usually have their own node/edge features and they do not provide SMILES
MOLECULAR_DATASETS = [
    "QM9",
    "DOCKSTRING",
    "ESOL",
    "FreeSolv",
    "Lipo",
    "PCBA",
    "MUV",
    "HIV",
    "BACE",
    "BBBP",
    "Tox21",
    "ToxCast",
    "SIDER",
    "ClinTox",
    "ZINC",
    "lrgb-pept-fn",
    "lrgn-pept-struct",
]


def convert_to_single_emb_first_then_same(x, first_offset: int = 120, other_offset: int = 10):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offsets = np.zeros(feature_num, dtype=np.int32)
    feature_offsets[0] = 1  # Start with an offset for the first feature
    feature_offsets[1:] = first_offset + (np.arange(1, feature_num) * other_offset)
    x = x + feature_offsets
    return x


def convert_to_single_emb_node(x, dataset_name, max_degree):
    # # For ChemProp features, we want only the first feature (atom type) to have a large offset
    if dataset_name in MOLECULAR_DATASETS:
        input_nodes = convert_to_single_emb_first_then_same(x, first_offset=120, other_offset=10) + 1
    else:
        # Otherwise we most likely have a degree integer and we must use it as a large offset
        if max_degree is not None:
            input_nodes = convert_to_single_emb(x, offset=max_degree) + 1
        # Otherwise we most likely have pre-defined one-hot node features. Add a small offset
        else:
            input_nodes = convert_to_single_emb(x, offset=2) + 1

    return input_nodes


def preprocess_item(item, max_items, overall_max_dist, is_node_task=False):
    requires_backends(preprocess_item, ["cython"])

    dataset_name = item["dataset_name"][0]
    max_degree = item["max_degree"] if "max_degree" in item else None
    if max_degree is not None:
        max_degree = torch.max(torch.tensor(max_degree).flatten()).item()

    node_feature = item["node_feat"].numpy()
    edge_index = item["edge_index"].numpy()
    if "edge_attr" in item.keys():
        edge_attr = item["edge_attr"].numpy()
    else:
        edge_attr = np.ones((edge_index.shape[-1], 1))

    input_nodes = convert_to_single_emb_node(node_feature, dataset_name, max_degree)
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]

    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int32)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr, offset=5)

    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)

    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    if not is_node_task:
        attn_bias = np.zeros(shape=[max_items + 1, max_items + 1], dtype=np.single)
    else:
        attn_bias = np.zeros(shape=[max_items, max_items], dtype=np.single)

    try:
        max_input_edges = np.max(input_edges)
    except:
        max_input_edges = np.max(np.array([0], dtype=np.int32))

    node_feat_size = input_nodes.shape[-1]
    edge_feat_size = edge_attr.shape[-1]
    
    attn_edge_type_pad = np.zeros((max_items, max_items, edge_feat_size), dtype=np.int32)
    spatial_pos_pad = np.zeros((max_items, max_items), dtype=np.int32)
    in_degree_pad = np.zeros((max_items,), dtype=np.int32)
    input_nodes_pad = np.zeros((max_items, node_feat_size), dtype=np.int32)
    input_edges_pad = np.zeros((max_items, max_items, overall_max_dist + 1, edge_feat_size), dtype=np.int32)

    input_nodes_pad[:num_nodes, :] = input_nodes + 1
    attn_edge_type_pad[:num_nodes, :num_nodes, :] = attn_edge_type + 1
    spatial_pos_pad[:num_nodes, :num_nodes] = shortest_path_result.astype(np.int32) + 1
    in_degree_pad[:num_nodes] = np.sum(adj, axis=1).reshape(-1) + 1

    max_dist_lim = max_dist if max_dist != 510 else overall_max_dist

    input_edges_pad[:num_nodes, :num_nodes, :max_dist_lim, :] = input_edges[:num_nodes, :num_nodes, :max_dist_lim, :] + 1

    item["input_nodes"] = torch.from_numpy(input_nodes_pad)
    item["attn_bias"] = torch.from_numpy(attn_bias)
    item["attn_edge_type"] = torch.from_numpy(attn_edge_type_pad)
    item["spatial_pos"] = torch.from_numpy(spatial_pos_pad)
    item["in_degree"] = torch.from_numpy(in_degree_pad)
    item["out_degree"] = torch.clone(item["in_degree"])
    item["input_edges"] = torch.from_numpy(input_edges_pad) 
    item["max_input_edges"] = torch.tensor(max_input_edges)
    item["labels"] = torch.clone(item["y"])
    item["max_dist"] = torch.tensor(max_dist)

    del item["y"]
    del item["edge_index"]
    del item["node_feat"]
    del item["dataset_name"]
    if "edge_attr" in item.keys():
        del item["edge_attr"]

    return item


class GraphormerDataCollator:
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        batch = {}

        batch["attn_edge_type"] = torch.stack([i["attn_edge_type"] for i in features])
        batch["spatial_pos"] = torch.stack([i["spatial_pos"] for i in features])
        batch["in_degree"] = torch.stack([i["in_degree"] for i in features])
        batch["input_nodes"] = torch.stack([i["input_nodes"] for i in features])
        batch["input_edges"] = torch.stack([i["input_edges"] for i in features])
        batch["attn_bias"] = torch.stack([i["attn_bias"] for i in features])
        batch["out_degree"] = batch["in_degree"]

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
            else:  # binary classification
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = torch.from_numpy(np.stack([i["labels"] for i in features], axis=0))

        return batch