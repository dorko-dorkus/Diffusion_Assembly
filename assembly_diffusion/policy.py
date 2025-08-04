import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .backbone import GNNBackbone
from .graph import MoleculeGraph


class ReversePolicy(nn.Module):
    """Reverse diffusion policy producing edit logits."""

    def __init__(self, backbone: GNNBackbone):
        super().__init__()
        self.backbone = backbone
        # Head predicting scores for bond orders 0, 1, 2 or 3
        self.edit_head = nn.Linear(backbone.node_dim * 2 + backbone.edge_dim, 4)
        # Separate head for the stop action
        self.stop_head = nn.Linear(backbone.node_dim, 1)
        # Stores mapping from logits to semantic edits for sampling
        self._actions = []

    def logits(self, x: MoleculeGraph, t: int, mask):
        """Compute logits for all possible edits including STOP.

        ``mask`` is a dictionary mapping edit tuples ``(i, j, b)`` or the
        string ``"STOP"`` to ``0/1`` feasibility flags.  Infeasible edits are
        assigned ``-inf`` logit scores so they are never sampled.
        """

        h_nodes, h_edges = self.backbone(x, t)

        scores = []
        actions = []
        for key, feasible in mask.items():
            if key == "STOP":
                continue
            i, j, b = key
            feat = torch.cat([h_nodes[i], h_nodes[j], h_edges[i, j]], dim=-1)
            logit = self.edit_head(feat)[b]
            if not feasible:
                logit = torch.tensor(-torch.inf, device=logit.device)
            scores.append(logit)
            actions.append(key)

        stop_score = self.stop_head(h_nodes.mean(dim=0)).squeeze()
        if not mask.get("STOP", 1):
            stop_score = torch.tensor(-torch.inf, device=stop_score.device)
        scores.append(stop_score)
        actions.append("STOP")

        self._actions = actions
        return torch.stack(scores)

    def sample_edit(self, logits: torch.Tensor, temperature: float = 1.0):
        """Sample an edit index and its semantic meaning.

        Parameters
        ----------
        logits: ``torch.Tensor``
            Logits returned by :meth:`logits`.
        temperature: ``float``, optional
            Softmax temperature used for sampling.
        """

        probs = torch.softmax(logits / temperature, dim=0)
        dist = Categorical(probs)
        idx = dist.sample()
        action = self._actions[idx.item()]
        return idx.item(), action

    def loss(self, logits: torch.Tensor, target: int, lambda_reg: float = 0.0):
        """Compute cross-entropy loss with optional entropy regularization."""

        target_tensor = torch.tensor([target], device=logits.device)
        ce = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        if lambda_reg > 0:
            probs = torch.softmax(logits, dim=0)
            entropy = -(probs * torch.log(probs + 1e-12)).sum()
            ce = ce - lambda_reg * entropy
        return ce

