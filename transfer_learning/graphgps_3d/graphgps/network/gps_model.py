import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder

from graphgps.layer.gps_layer import GPSLayer

from torch_geometric.utils import to_dense_batch, unbatch_edge_index

import sys
sys.path.append('/oscache/cc/kfjs289/graphgps_3d/')

from data_loading.gaussian import GaussianLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        print('GPS model dim_in = ', dim_in)
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.atom_types = 10
        self.edge_types = 10 * 10
        self.k = 32
        self.embed_dim = 32
        self.atom_encoder = nn.Embedding(self.atom_types, self.embed_dim, padding_idx=0)
        self.edge_proj = nn.Linear(self.k, self.embed_dim)
        self.gbf = GaussianLayer(self.k, self.edge_types)

        self.proj_in_nodes = nn.Linear(self.embed_dim, cfg.gt.dim_hidden)
        self.proj_in_edges = nn.Linear(self.embed_dim, cfg.gt.dim_hidden)


    def forward(self, batch):
        z = batch.z
        pos = batch.pos
        batch_mapping = batch.batch
        edge_index_radius = batch.edge_index

        z_unbatched, batch_mask = to_dense_batch(z, batch=batch_mapping, max_num_nodes=30, fill_value=0)
        pos_unbatched, _ = to_dense_batch(pos, batch=batch_mapping, max_num_nodes=30, fill_value=0)

        n_graph, n_node = z_unbatched.size()

        delta_pos = pos_unbatched.unsqueeze(1) - pos_unbatched.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)

        edge_type = z_unbatched.view(n_graph, n_node, 1) * self.atom_types + z_unbatched.view(n_graph, 1, n_node)
        gbf_feature = self.gbf(dist, edge_type)

        padding_mask = z_unbatched.eq(0)

        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.atom_encoder(z_unbatched)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        edge_index_radius = edge_index_radius[[1, 0], :]

        graph_node_feature_batched = graph_node_feature[batch_mask]

        edge_batch_mapping = batch_mapping.index_select(0, edge_index_radius[0, :])
        edge_index_unbatched = unbatch_edge_index(edge_index_radius, batch_mapping)
        edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

        edge_features_source_target = edge_features[edge_batch_mapping, edge_batch_non_cumulative[0, :], edge_batch_non_cumulative[1, :], :]

        node_feat = self.proj_in_nodes(graph_node_feature_batched)
        edge_feat = self.proj_in_edges(edge_features_source_target)

        batch.x = node_feat
        batch.edge_attr = edge_feat

        # print('self.children = ', list(self.children())[:-4])

        for module in list(self.children())[:-5]:
            batch = module(batch)
        return batch
