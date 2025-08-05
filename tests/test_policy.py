import pathlib
import sys

import pytest

# Import torch lazily to allow the remaining tests to run when the
# dependency is unavailable.
torch = pytest.importorskip("torch")
import torch.nn as nn

# Ensure repository root on path for direct import
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.policy import ReversePolicy


class DummyBackbone(nn.Module):
    def __init__(self, B=2, N=3, node_dim=8, edge_dim=4):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_nodes = torch.randn(B, N, node_dim)
        self.h_edges = torch.randn(B, N, N, edge_dim)

    def forward(self, x, t):  # pragma: no cover - x is ignored
        return self.h_nodes, self.h_edges


def test_logits_batched_mask():
    backbone = DummyBackbone()
    policy = ReversePolicy(backbone)

    mask = {
        (0, 1, 0): torch.tensor([1, 0]),
        (0, 2, 1): torch.tensor([1, 1]),
        "STOP": torch.tensor([1, 0]),
    }

    logits = policy.logits(None, 0, mask)
    assert logits.shape == (2, len(policy._actions))

    idx_edit = policy._actions.index((0, 1, 0))
    assert logits[1, idx_edit] == float("-inf")

    idx_stop = policy._actions.index("STOP")
    assert logits[1, idx_stop] == float("-inf")

    # All logits for first batch element should be finite
    assert torch.isfinite(logits[0]).all()
