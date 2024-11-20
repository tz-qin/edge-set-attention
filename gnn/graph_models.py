import os
import torch
import torch_geometric
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb


from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    GATConv,
    GATv2Conv,
    GINConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.utils import degree
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional

from utils.norm_layers import BN
from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)


def nearest_multiple_of_five(n):
    return round(n / 5) * 5


def get_degrees(train_dataset_as_list, out_path):
    pna_degrees_save_dir_path = os.path.join(out_path, "saved_PNA_degrees")
    pna_degrees_save_file_path = os.path.join(out_path, "saved_PNA_degrees", "degrees.pt")

    if Path(pna_degrees_save_file_path).is_file():
        print("Loaded degrees for PNA from saved file!")
        deg = torch.load(pna_degrees_save_file_path)
    else:
        deg = torch.zeros(5000, dtype=torch.long)
        print("Computing degrees for PNA...")
        for data in tqdm(train_dataset_as_list):
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

    Path(pna_degrees_save_dir_path).mkdir(exist_ok=True, parents=True)
    torch.save(deg, pna_degrees_save_file_path)

    return deg



# GNN layers with skip connection

class GCNConvSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvSC, self).__init__()

        self.gnn_layer = GCNConv(in_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        return x + self.gnn_layer(x, edge_index)
    

class GINConvSC(nn.Module):
    def __init__(self, GINConvConstructor, in_channels, out_channels, edge_dim):
        super(GINConvSC, self).__init__()

        if edge_dim is not None:
            self.gnn_layer = GINConvConstructor(
                                nn.Sequential(
                                    Linear(in_channels, in_channels),
                                    nn.Mish(),
                                    Linear(in_channels, out_channels),
                                ),
                                edge_dim=edge_dim,
                            )
        else:
            self.gnn_layer = GINConvConstructor(
                                nn.Sequential(
                                    Linear(in_channels, in_channels),
                                    nn.Mish(),
                                    Linear(in_channels, out_channels),
                                ),
                            )

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            return x + self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        else:
            return x + self.gnn_layer(x, edge_index)


class PNAConvSC(nn.Module):
    def __init__(self, in_channels, out_channels, **pna_args):
        super(PNAConvSC, self).__init__()

        self.gnn_layer = PNAConv(in_channels=in_channels, out_channels=out_channels, **pna_args)

    def forward(self, x, edge_index, edge_attr=None):
        return x + self.gnn_layer(x, edge_index, edge_attr=edge_attr)
    

class GATorGATv2SC(nn.Module):
    def __init__(self, GATConvConstructor, in_channels, out_channels, attn_heads, concat, dropout, edge_dim):
        super(GATorGATv2SC, self).__init__()

        self.gnn_layer = GATConvConstructor(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            heads=attn_heads,
                            concat=concat,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        )

    def forward(self, x, edge_index, edge_attr=None):
        try:
            return x + self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        except:
            return self.gnn_layer(x, edge_index, edge_attr=edge_attr)


# Code adapted from https://github.com/KarolisMart/DropGNN
class GINDrop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_runs: int,
        p: float,
        edge_dim: int = None,
    ):
        
        super(GINDrop, self).__init__()
        self.num_runs = num_runs
        self.p = p
        self.edge_dim = edge_dim

        if self.edge_dim is not None:
            self.conv = GINEConv(
                nn.Sequential(
                    Linear(in_channels, intermediate_dim),
                    nn.Mish(),
                    Linear(intermediate_dim, out_channels),
                ),
                edge_dim=self.edge_dim,
            )
        else:
            self.conv = GINConv(
                nn.Sequential(
                    Linear(in_channels, intermediate_dim),
                    nn.Mish(),
                    Linear(intermediate_dim, out_channels),
                )
            )

    def forward(self, x, edge_index, edge_attr=None):
        x_original = x
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()

        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device, dtype=x.dtype) * self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device, dtype=x.dtype)
        del drop

        x = x.view(-1, x.size(-1))
        
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
        if edge_attr is not None:
            run_edge_attr = edge_attr.repeat(self.num_runs, 1)
            m = self.conv(x, run_edge_index, run_edge_attr)
            del run_edge_attr
        else:
            m = self.conv(x, run_edge_index)
        
        del run_edge_index

        m = m.view(self.num_runs, -1, m.size(-1))
        m = m.mean(dim=0)

        return m + x_original

# GNN layers with skip connection

# ############# GNN modules ##############

class GCN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GCN, self).__init__()

        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        modules = []

        for i in range(num_layers):
            if i == 0:
                modules.append((GCNConvSC(intermediate_dim, intermediate_dim), "x, edge_index -> x"))

                modules.append(BN(intermediate_dim))
            elif i != num_layers - 1:
                modules.append((GCNConvSC(intermediate_dim, intermediate_dim), "x, edge_index -> x"))

                modules.append(BN(intermediate_dim))
            elif i == num_layers - 1:
                modules.append((GCNConvSC(intermediate_dim, out_channels), "x, edge_index -> x"))

                modules.append(BN(out_channels))
            modules.append(nn.Mish())

        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.input_mlp(x)
        x_conv = self.convs(x_proj, edge_index=edge_index)

        return x_proj + x_conv


class GIN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GIN, self).__init__()
        self.edge_dim = edge_dim

        modules = []

        self.node_input_mlp = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        if self.edge_dim is not None:
            self.edge_input_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim, bias=True),
                nn.Mish(),
                nn.Linear(edge_dim, edge_dim, bias=True),
            )

        GINConvConstructor = GINEConv if edge_dim else GINConv
        seq_str = "x, edge_index, edge_attr -> x" if edge_dim else "x, edge_index -> x"

        for i in range(num_layers):
            if i == 0:
                modules.append(
                    (
                        GINConvSC(GINConvConstructor, intermediate_dim, intermediate_dim, edge_dim),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim))

            elif i != num_layers - 1:
                modules.append(
                    (
                        GINConvSC(GINConvConstructor, intermediate_dim, intermediate_dim, edge_dim),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim))

            elif i == num_layers - 1:
                modules.append(
                    (
                        GINConvSC(GINConvConstructor, intermediate_dim, out_channels, edge_dim),
                        seq_str,
                    )
                )

                modules.append(BN(out_channels))

            modules.append(nn.Mish())

        if edge_dim:
            self.convs = torch_geometric.nn.Sequential("x, edge_index, edge_attr", modules)
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.node_input_mlp(x)
        if edge_attr is not None:
            edge_attr_proj = self.edge_input_mlp(edge_attr)

        if edge_attr is not None:
            x_conv = self.convs(x_proj, edge_index, edge_attr=edge_attr_proj)
        else:
            x_conv = self.convs(x_proj, edge_index)

        return x_proj + x_conv


class PNA(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        train_dataset,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(PNA, self).__init__()
        self.edge_dim = edge_dim

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = get_degrees(train_dataset, out_path=out_path)

        print("Adjusting graph hidden dimensions so that they are divisible by the number of PNA towers (5).")
        self.intermediate_dim = nearest_multiple_of_five(intermediate_dim)
        self.out_channels = nearest_multiple_of_five(out_channels)
        self.original_out_channels = out_channels

        self.node_input_mlp = nn.Sequential(
            nn.Linear(in_channels, self.intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim, bias=True),
        )

        if self.edge_dim is not None:
            self.edge_input_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim, bias=True),
                nn.Mish(),
                nn.Linear(edge_dim, edge_dim, bias=True),
            )

        self.final_proj = nn.Linear(self.out_channels, self.original_out_channels)

        pna_num_towers = 5

        pna_common_args = dict(
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=None,
            towers=pna_num_towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        if edge_dim:
            pna_common_args = pna_common_args | dict(edge_dim=edge_dim)

        modules = []

        seq_str = "x, edge_index, edge_attr -> x" if edge_dim else "x, edge_index -> x"

        for i in range(num_layers):
            if i == 0:
                modules.append(
                    (
                        PNAConvSC(self.intermediate_dim, self.intermediate_dim, **pna_common_args),
                        seq_str,
                    )
                )

                modules.append(BN(self.intermediate_dim))

            elif i != num_layers - 1:
                modules.append(
                    (
                        PNAConvSC(self.intermediate_dim, self.intermediate_dim, **pna_common_args),
                        seq_str,
                    )
                )

                modules.append(BN(self.intermediate_dim))

            elif i == num_layers - 1:
                modules.append(
                    (
                        PNAConvSC(self.intermediate_dim, self.out_channels, **pna_common_args),
                        seq_str,
                    )
                )

                modules.append(BN(self.out_channels))

            modules.append(nn.Mish())

        if edge_dim:
            self.convs = torch_geometric.nn.Sequential("x, edge_index, edge_attr", modules)
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.node_input_mlp(x)
        if edge_attr is not None:
            edge_attr_proj = self.edge_input_mlp(edge_attr)

        if edge_attr is not None:
            x_conv = self.convs(x_proj, edge_index, edge_attr=edge_attr_proj)
        else:
            x_conv = self.convs(x_proj, edge_index)

        return self.final_proj(x_proj + x_conv)


class GATorGATv2(pl.LightningModule):
    def __init__(
        self,
        gat_or_gatv2: str,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        attn_heads: int,
        dropout: float,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GATorGATv2, self).__init__()
        self.edge_dim = edge_dim

        assert gat_or_gatv2 in ["GAT", "GATv2"]
        GATConvConstructor = GATConv if gat_or_gatv2 == "GAT" else GATv2Conv

        self.node_input_mlp = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        if self.edge_dim is not None:
            self.edge_input_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim, bias=True),
                nn.Mish(),
                nn.Linear(edge_dim, edge_dim, bias=True),
            )

        modules = []

        seq_str = "x, edge_index, edge_attr -> x" if edge_dim else "x, edge_index -> x"

        for i in range(num_layers):
            if i == 0:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=True,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim * attn_heads))

            elif i != num_layers - 1:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim * attn_heads,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=True,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim * attn_heads))

            else:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim * attn_heads,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=False,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(out_channels))

            modules.append(nn.Mish())

        if edge_dim:
            self.convs = torch_geometric.nn.Sequential("x, edge_index, edge_attr", modules)
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.node_input_mlp(x)
        if edge_attr is not None:
            edge_attr_proj = self.edge_input_mlp(edge_attr)

        if edge_attr is not None:
            x_conv = self.convs(x_proj, edge_index, edge_attr=edge_attr_proj)
        else:
            x_conv = self.convs(x_proj, edge_index)

        return x_proj + x_conv


class GINDropEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        num_runs: int,
        p: float,
        edge_dim: int = None,
        out_path: str=None,
    ):
        super(GINDropEncoder, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        self.node_input_mlp = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        if self.edge_dim is not None:
            self.edge_input_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim, bias=True),
                nn.Mish(),
                nn.Linear(edge_dim, edge_dim, bias=True),
            )

        modules = []

        seq_str = "x, edge_index, edge_attr -> x" if edge_dim else "x, edge_index -> x"

        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                        (
                            GINDrop(in_channels=intermediate_dim, intermediate_dim=intermediate_dim, out_channels=intermediate_dim, edge_dim=edge_dim, num_runs=num_runs, p=p),
                            seq_str,
                        )
                    )
                modules.append(BN(intermediate_dim))
            elif i != num_layers - 1:
                modules.append(
                        (
                            GINDrop(in_channels=intermediate_dim, intermediate_dim=intermediate_dim, out_channels=intermediate_dim, edge_dim=edge_dim, num_runs=num_runs, p=p),
                            seq_str,
                        )
                    )
                modules.append(BN(intermediate_dim))
            else:
                modules.append(
                        (
                            GINDrop(in_channels=intermediate_dim, intermediate_dim=intermediate_dim, out_channels=out_channels, edge_dim=edge_dim, num_runs=num_runs, p=p),
                            seq_str,
                        )
                    )
                modules.append(BN(out_channels))

            modules.append(nn.Mish())

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential("x, edge_index, edge_attr", modules)
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)


    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.node_input_mlp(x)
        if edge_attr is not None:
            edge_attr_proj = self.edge_input_mlp(edge_attr)

        if edge_attr is not None:
            x_conv = self.convs(x_proj, edge_index, edge_attr=edge_attr_proj)
        else:
            x_conv = self.convs(x_proj, edge_index)

        return x_proj + x_conv

# ############# GNN modules ##############


class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        gnn_intermediate_dim: int,
        output_node_dim: int,
        batch_size: int = 32,
        lr: float = 0.001,
        conv_type: str = "GCN",
        gat_attn_heads: int = 4,
        gat_dropout: float = 0,
        linear_output_size: int = 1,
        output_intermediate_dim: int = 768,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        num_layers: int = None,
        edge_dim: int = None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        out_path: str = None,
        train_dataset_for_PNA=None,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        use_cpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert task_type in ["binary_classification", "multi_classification", "regression"]
        assert conv_type in ["GCN", "GIN", "PNA", "GAT", "GATv2", "GINDrop"]

        self.edge_dim = edge_dim
        self.task_type = task_type
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.conv_type = conv_type
        self.output_node_dim = output_node_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.output_intermediate_dim = output_intermediate_dim
        self.num_layers = num_layers
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss_name = monitor_loss_name
        self.out_path = out_path
        self.regression_loss_fn = regression_loss_fn
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.gat_attn_heads = gat_attn_heads
        self.gat_dropout = gat_dropout
        self.use_cpu = use_cpu

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        # Node task masks
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        gnn_args = dict(
            in_channels=num_features,
            out_channels=output_node_dim,
            intermediate_dim=gnn_intermediate_dim,
            num_layers=num_layers,
            out_path=out_path,
        )
        if self.edge_dim:
            gnn_args = gnn_args | dict(edge_dim=edge_dim)
        if self.conv_type == "PNA":
            gnn_args = gnn_args | dict(train_dataset=train_dataset_for_PNA)
        if self.conv_type in ["GAT", "GATv2"]:
            gnn_args = gnn_args | dict(attn_heads=gat_attn_heads, dropout=gat_dropout)
        if self.conv_type in ["GINDrop"]:
            gnn_args = gnn_args | dict(p=0.2, num_runs=40, use_batch_norm=True)

        if self.conv_type == "GCN":
            self.gnn_model = GCN(**gnn_args)
        elif self.conv_type == "GIN":
            self.gnn_model = GIN(**gnn_args)
        elif self.conv_type == "PNA":
            self.gnn_model = PNA(**gnn_args)
        elif self.conv_type == "GAT":
            self.gnn_model = GATorGATv2(gat_or_gatv2="GAT", **gnn_args)
        elif self.conv_type == "GATv2":
            self.gnn_model = GATorGATv2(gat_or_gatv2="GATv2", **gnn_args)
        elif self.conv_type == "GINDrop":
            self.gnn_model = GINDropEncoder(**gnn_args)

        if self.train_mask is None:
            output_mlp_in_dim = output_node_dim * 3
        else:
            output_mlp_in_dim = output_node_dim

        self.output_mlp = nn.Sequential(
            nn.Linear(output_mlp_in_dim, 64), nn.BatchNorm1d(64), nn.Mish(), nn.Linear(64, linear_output_size)
        )

        if self.conv_type == "PNA":
            output_node_dim = nearest_multiple_of_five(output_node_dim)


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # 1. Obtain node embeddings
        if self.edge_dim and self.conv_type != "GCN":
            z = self.gnn_model.forward(x, edge_index, edge_attr=edge_attr)
        else:
            z = self.gnn_model.forward(x, edge_index)

        if self.train_mask is None:
            # 2. Readout layer (sumple global pooling of node features)
            emb_sum_pool = global_add_pool(z, batch)
            emb_avg_pool = global_mean_pool(z, batch)
            emb_max_pool = global_max_pool(z, batch)

            global_emb_pool = torch.cat((emb_sum_pool, emb_avg_pool, emb_max_pool), dim=-1)
            gnn_out = global_emb_pool
        else:
            global_emb_pool = None
            gnn_out = z

        # 3. Apply a final classifier
        predictions = torch.flatten(self.output_mlp(gnn_out))

        return z, global_emb_pool, predictions

    def configure_optimizers(self):
        if not self.use_cpu:
            opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        self.monitor_loss_name = "Validation MCC" if "MCC" in self.monitor_loss_name or self.monitor_loss_name == "MCC" else self.monitor_loss_name
        mode = "max" if "MCC" in self.monitor_loss_name else "min"

        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        if self.monitor_loss_name != "train_loss":
            opt_dict["lr_scheduler"] = sched

        return opt_dict
    

    def _batch_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        batch_mapping: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        step_type: str = None,
    ):
        # Forward pass (graph_embeddings not used here so far, after forward)
        z, graph_embeddings, predictions = self.forward(x, edge_index, batch_mapping, edge_attr=edge_attr)

        if self.task_type == "multi_classification":
            predictions = predictions.reshape(-1, self.linear_output_size)

            predictions = predictions.squeeze().float()
            y = y.squeeze().long()

            if step_type == "train" and self.train_mask is not None:
                predictions = predictions[self.train_mask]
                y = y[self.train_mask]
            
            if step_type == "validation" and self.val_mask is not None:
                predictions = predictions[self.val_mask]
                y = y[self.val_mask]

            if step_type == "test" and self.test_mask is not None:
                predictions = predictions[self.test_mask]
                y = y[self.test_mask]

            task_loss = F.cross_entropy(predictions.squeeze().float(), y.squeeze().long())

        elif self.task_type == "binary_classification":
            y = y.view(predictions.shape)
            task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())

        else:
            if self.regression_loss_fn == "mse":
                task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))
            elif self.regression_loss_fn == "mae":
                task_loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y.float()))

        return task_loss, z, graph_embeddings, predictions, y


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        x, edge_index, y, batch_mapping, edge_attr = batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr

        total_loss, z, graph_embeddings, predictions, y = self._batch_loss(
            x, edge_index, y, batch_mapping, edge_attr=edge_attr, step_type=step_type,
        )

        if self.task_type == "regression":
            output = (torch.flatten(predictions), torch.flatten(y))
        elif "classification" in self.task_type:
            output = (predictions, y)

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_loss = self._step(batch, "train")

        self.log("train_loss", train_loss, prog_bar=True, batch_size=self.batch_size)

        return train_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_loss = self._step(batch, "validation")

            self.log("val_loss", val_loss, batch_size=self.batch_size)

            return val_loss

        if dataloader_idx == 1:
            val_test_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_loss, batch_size=self.batch_size)

            return val_test_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_loss = self._step(batch, "test")

        self.log("test_loss", test_loss, batch_size=self.batch_size)

        return test_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        assert epoch_type in ["Train", "Validation", "Test", "ValidationTest"]

        def flatten_list_of_tensors(lst):
            return np.array([item.item() for sublist in lst for item in sublist])

        if self.task_type == "regression":
            y_pred, y_true = flatten_list_of_tensors([item[0] for item in epoch_outputs]), flatten_list_of_tensors(
                [item[1] for item in epoch_outputs]
            )
        else:
            y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
            y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)

        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)

        if self.task_type == "binary_classification" and self.linear_output_size > 1:
            y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
            y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)

            metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "binary_classification" and self.linear_output_size == 1:
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "multi_classification" and self.linear_output_size > 1:
            metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "regression":
            metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())

            self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
            self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)

        return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1
