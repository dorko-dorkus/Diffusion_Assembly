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

        Parameters
        ----------
        x:
            Input molecular graph. When ``x`` represents a batch of graphs the
            backbone is expected to return node features with shape
            ``(B, N, D)`` and edge features with shape ``(B, N, N, E)``.  For a
            single graph the shapes reduce to ``(N, D)`` and ``(N, N, E)``.
        t:
            Diffusion time step.
        mask:
            Dictionary mapping edit tuples ``(i, j, b)`` or the string
            ``"STOP"`` to feasibility flags.  In the batched case the flags
            should be tensors of shape ``(B,)``.  Infeasible edits are assigned
            ``-inf`` logit scores so they are never sampled.
        """

        h_nodes, h_edges = self.backbone(x, t)

        # Normalize shapes so that ``h_nodes`` has shape ``(B, N, D)`` and
        # ``h_edges`` has shape ``(B, N, N, E)`` regardless of whether ``x``
        # represents a single molecule or a batch.
        batched = h_nodes.dim() == 3
        if not batched:
            h_nodes = h_nodes.unsqueeze(0)
            h_edges = h_edges.unsqueeze(0)
        B, N, _ = h_nodes.shape

        scores = []
        actions = []
        for key, feasible in mask.items():
            if key == "STOP":
                continue
            i, j, b = key
            feat = torch.cat(
                [h_nodes[:, i], h_nodes[:, j], h_edges[:, i, j]], dim=-1
            )
            logit = self.edit_head(feat)[:, b]
            feas = torch.as_tensor(feasible, device=logit.device)
            if feas.ndim == 0:
                feas = feas.expand(B)
            logit = torch.where(feas.bool(), logit, torch.full_like(logit, -torch.inf))
            scores.append(logit)
            actions.append(key)

        stop_score = self.stop_head(h_nodes.mean(dim=1)).squeeze(-1)
        stop_feas = mask.get("STOP", torch.ones(B, device=stop_score.device))
        stop_feas = torch.as_tensor(stop_feas, device=stop_score.device)
        if stop_feas.ndim == 0:
            stop_feas = stop_feas.expand(B)
        stop_score = torch.where(
            stop_feas.bool(), stop_score, torch.full_like(stop_score, -torch.inf)
        )
        scores.append(stop_score)
        actions.append("STOP")

        self._actions = actions
        logits = torch.stack(scores, dim=1)
        return logits[0] if not batched else logits

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

