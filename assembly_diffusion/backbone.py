import torch
import torch.nn as nn

from .graph import MoleculeGraph

class GNNBackbone(nn.Module):
    """Minimal GNN backbone used by the reverse policy."""
    def __init__(self, node_dim: int = 16):
        super().__init__()
        self.node_fc = nn.Linear(node_dim, node_dim)

    def forward(self, x: MoleculeGraph, t: int):
        h_nodes = torch.randn(len(x.atoms), 16)
        h_edges = torch.randn(len(x.atoms), len(x.atoms), 4)
        return h_nodes, h_edges
