import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            self.in_dim = 1
        elif cfg.dataset.name == "ogbn-proteins":
            self.in_dim = 8
        elif cfg.dataset.name in ['QM9', 'DOCKSTRING', 'ESOL', 'FreeSolv', 'Lipo', 'PCBA', 'MUV', 'HIV', 'BACE', 'BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox']:
            self.in_dim = 13
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)
        # self.encoder = torch.nn.Linear(self.in_dim, cfg.gt.dim_hidden)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim).float())
        return batch
