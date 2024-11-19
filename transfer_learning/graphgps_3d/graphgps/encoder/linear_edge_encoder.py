import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.in_dim = emb_dim
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)
        # self.encoder = torch.nn.Linear(self.in_dim, cfg.gt.dim_hidden)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim).float())
        return batch
