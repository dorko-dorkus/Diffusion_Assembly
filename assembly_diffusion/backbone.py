import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import MoleculeGraph


# Small list of common atom types used for one-hot encoding.  Unknown atoms are
# mapped to the first index.
ATOM_TYPES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_MAP = {a: i for i, a in enumerate(ATOM_TYPES)}


class GraphConv(nn.Module):
    """Simple graph convolution using adjacency matrices."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # ``adj`` is expected to be an ``(N, N)`` matrix with edge weights.  We
        # aggregate neighbour messages using a linear projection and multiply by
        # the adjacency matrix to weight contributions by bond order.
        neigh = torch.matmul(adj, self.neigh_lin(h))
        return self.self_lin(h) + neigh


def time_embedding(t: int, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding :math:`\\psi(t)` of dimension ``dim``."""

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
    )
    ang = t * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class GNNBackbone(nn.Module):
    """Minimal GNN backbone used by the reverse policy."""

    def __init__(self, node_dim: int = 16, edge_dim: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Node/edge encoders
        node_feat_dim = len(ATOM_TYPES) + 5  # atom type + degree one-hot
        self.node_encoder = nn.Linear(node_feat_dim, node_dim)

        self.conv1 = GraphConv(node_dim, node_dim)
        self.conv2 = GraphConv(node_dim, node_dim)

        edge_input_dim = node_dim * 2 + 4  # two nodes + bond order one-hot
        self.edge_mlp = nn.Linear(edge_input_dim, edge_dim)

    def forward(self, x: MoleculeGraph, t: int):
        n = len(x.atoms)

        # --- Node features -------------------------------------------------
        atom_ids = torch.tensor(
            [ATOM_MAP.get(a, 0) for a in x.atoms], device=x.bonds.device
        )
        atom_feat = F.one_hot(atom_ids, num_classes=len(ATOM_TYPES)).float()

        degree = (x.bonds > 0).sum(dim=1).clamp(max=4)
        deg_feat = F.one_hot(degree, num_classes=5).float()

        node_feat = torch.cat([atom_feat, deg_feat], dim=-1)
        h = self.node_encoder(node_feat)

        # --- Time embedding ------------------------------------------------
        t_emb = time_embedding(t, self.node_dim).to(h.device)

        # --- Message passing -----------------------------------------------
        adj = x.bonds.float()
        h = self.conv1(h + t_emb, adj)
        h = F.relu(h)
        h = self.conv2(h + t_emb, adj)
        h_nodes = h

        # --- Edge features -------------------------------------------------
        bond_feat = F.one_hot(x.bonds.clamp(min=0, max=3).long(), num_classes=4).float()
        hi = h.unsqueeze(1).expand(-1, n, -1)
        hj = h.unsqueeze(0).expand(n, -1, -1)
        edge_input = torch.cat([hi, hj, bond_feat], dim=-1)
        h_edges = self.edge_mlp(edge_input)

        return h_nodes, h_edges
