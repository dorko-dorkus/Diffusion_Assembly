import pathlib
import torch
import torch.nn as nn
import sys

# Ensure repository root on path for direct import
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.policy import ReversePolicy
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.sampler import Sampler


class DummyBackbone(nn.Module):
    def __init__(self, N=2, node_dim=8, edge_dim=4):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_nodes = torch.randn(N, node_dim)
        self.h_edges = torch.randn(N, N, edge_dim)

    def forward(self, x, t):  # pragma: no cover - x is ignored
        return self.h_nodes, self.h_edges


class StopOnlyMask:
    def mask_edits(self, x):
        return {"STOP": 1}


def test_sampler_stops_early_when_stop_only():
    backbone = DummyBackbone()
    policy = ReversePolicy(backbone)
    sampler = Sampler(policy, StopOnlyMask())

    bonds = torch.zeros((2, 2), dtype=torch.int64)
    x_init = MoleculeGraph(["C", "C"], bonds)

    result = sampler.sample(T=5, x_init=x_init)
    assert torch.equal(result.bonds, x_init.bonds)
