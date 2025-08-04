import torch
import torch.nn as nn

from .backbone import GNNBackbone
from .graph import MoleculeGraph


class ReversePolicy(nn.Module):
    """Reverse diffusion policy producing edit logits."""
    def __init__(self, backbone: GNNBackbone):
        super().__init__()
        self.backbone = backbone
        self.stop_head = nn.Linear(16, 1)

    def logits(self, x: MoleculeGraph, t: int, mask):
        h_nodes, _ = self.backbone(x, t)
        e_scores = torch.randn(len(mask) - 1)
        stop_score = self.stop_head(h_nodes.mean(dim=0))
        return torch.cat([e_scores, stop_score.flatten()])
