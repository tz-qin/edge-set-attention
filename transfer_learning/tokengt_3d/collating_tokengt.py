# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

from typing import Any, Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from transformers.utils.import_utils import is_cython_available


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_tokengt  # noqa E402


# From algos.py
def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = EigVec.astype(dtype=np.single)  # [num_nodes, num_nodes (channels)]
    eigval = np.sort(np.abs(np.real(EigVal))).astype(dtype=np.single)  # [num_nodes (channels),]
    return eigvec, eigval  # [num_nodes, num_nodes (channels)]  [num_nodes (channels),]


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    # Laplacian
    A = dense_adj.astype(dtype=np.single)
    num_nodes = np.diag(in_degree.astype(dtype=np.single).clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - num_nodes @ A @ num_nodes

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [num_nodes, num_nodes (channels)]  [num_nodes (channels),]



def preprocess_labels_only(item, task_list=None):
    task_list = ["y"] if task_list is None else task_list
    item["labels"] = {}
    for task in task_list:
        if task in item.keys():
            item["labels"][task] = item[task]

    return item


def preprocess_item(item):
    max_degree = item["max_degree"] if "max_degree" in item else None
    if max_degree is not None:
        max_degree = torch.max(torch.tensor(max_degree).flatten()).item()

    edge_index = np.asarray(item["edge_index"], dtype=np.int32)
    pos = np.asarray(item["pos"], dtype=np.float32)
    z = np.asarray(item["z"], dtype=np.int32)

    num_nodes = item["num_nodes"]
    num_edges = item["num_edges"]
    dense_adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    dense_adj[edge_index[0], edge_index[1]] = True

    in_degree = np.sum(dense_adj, axis=1).reshape(-1)
    lap_eigvec, lap_eigval = lap_eig(dense_adj, num_nodes, in_degree)  # [num_nodes, num_nodes], [num_nodes,]
    lap_eigval = np.broadcast_to(lap_eigval[None, :], lap_eigvec.shape)

    # +1 are to shift indexes, for nn.Embedding with pad_index=0
    item["num_nodes"] = num_nodes
    item["num_edges"] = num_edges
    item["edge_index"] = edge_index
    item["in_degree"] = in_degree + 1
    item["out_degree"] = in_degree + 1  # for undirected graph, directed graphs not managed atm
    item["lap_eigvec"] = lap_eigvec
    item["lap_eigval"] = lap_eigval
    item["pos"] = pos
    item["z"] = z
    if "labels" not in item:
        item["labels"] = item["y"]  # default label tends to be y

    return item


class TokenGTDataCollator:
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        # self.tokenizer = tokenizer
        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    @torch.no_grad()
    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        batch = {}

        batch["num_nodes"] = torch.tensor([i["num_nodes"] for i in features])
        batch["num_edges"] = torch.tensor([i["num_edges"] for i in features])
        max_n = max(batch["num_nodes"])

        batch["edge_index"] = torch.cat(
            [torch.tensor(i["edge_index"], dtype=torch.long) for i in features], dim=1
        )  # [2, sum(edge_num)]
        batch["pos"] = torch.cat(
            [torch.tensor(i["pos"], dtype=torch.float32) for i in features], dim=0
        )
        batch["z"] = torch.cat(
            [torch.tensor(i["z"], dtype=torch.long) for i in features], dim=0
        )
        batch["in_degree"] = torch.cat(
            [torch.tensor(i["in_degree"], dtype=torch.long) for i in features]
        )  # [sum(node_num),],
        batch["out_degree"] = torch.cat(
            [torch.tensor(i["out_degree"], dtype=torch.long) for i in features]
        )  # [sum(node_num),],
        batch["lap_eigvec"] = torch.cat(
            [
                F.pad(
                    torch.tensor(i["lap_eigvec"], dtype=torch.float),
                    (0, max_n - len(i["lap_eigvec"][0])),
                    value=float("0"),
                )
                for i in features
            ]
        )
        batch["lap_eigval"] = torch.cat(
            [
                F.pad(
                    torch.tensor(i["lap_eigval"], dtype=torch.float),
                    (0, max_n - len(i["lap_eigval"][0])),
                    value=float("0"),
                )
                for i in features
            ]
        )

        batch["labels"] = {}
        sample = torch.tensor(features[0]["labels"]).squeeze()

        if sample.ndim == 0:
            sample = sample.unsqueeze(0)

        batch["labels"] = torch.cat(
            [torch.tensor(i["labels"], dtype=torch.float) for i in features]
        ) # [batch_size,]

        return batch
