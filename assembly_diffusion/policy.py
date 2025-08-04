import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .backbone import GNNBackbone
from .graph import MoleculeGraph, ALLOWED_ATOMS


class ReversePolicy(nn.Module):
    """Reverse diffusion policy producing edit logits."""

    def __init__(self, backbone: GNNBackbone):
        super().__init__()
        self.backbone = backbone
        # Head predicting scores for bond orders 0, 1, 2 or 3
        self.edit_head = nn.Linear(backbone.node_dim * 2 + backbone.edge_dim, 4)
        # Head for atom insertion actions (site + atom type)
        self.atom_embed = nn.Embedding(len(ALLOWED_ATOMS), backbone.node_dim)
        self.insert_head = nn.Linear(backbone.node_dim * 2, 1)
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

        device = h_nodes.device

        # Collect edit indices and feasibility flags
        edit_keys = [key for key in mask.keys() if isinstance(key, tuple) and key and key[0] != "ADD"]
        insert_keys = [key for key in mask.keys() if isinstance(key, tuple) and key and key[0] == "ADD"]
        actions = edit_keys + insert_keys + ["STOP"]

        if edit_keys:
            i_idx = torch.tensor([k[0] for k in edit_keys], device=device, dtype=torch.long)
            j_idx = torch.tensor([k[1] for k in edit_keys], device=device, dtype=torch.long)
            b_idx = torch.tensor([k[2] for k in edit_keys], device=device, dtype=torch.long)

            # Extract features for all candidate bond edits using advanced indexing
            node_i = h_nodes[:, i_idx]
            node_j = h_nodes[:, j_idx]
            edge_ij = h_edges[:, i_idx, j_idx]
            feat = torch.cat([node_i, node_j, edge_ij], dim=-1)

            edit_logits = self.edit_head(feat)
            edit_logits = edit_logits.gather(
                -1, b_idx.view(1, -1, 1).expand(B, -1, 1)
            ).squeeze(-1)

            feas_list = []
            for key in edit_keys:
                feas = torch.as_tensor(mask[key], device=device)
                if feas.ndim == 0:
                    feas = feas.expand(B)
                feas_list.append(feas)
            feas_tensor = torch.stack(feas_list, dim=1)
            edit_logits = torch.where(
                feas_tensor.bool(), edit_logits, torch.full_like(edit_logits, -torch.inf)
            )
        else:
            edit_logits = torch.empty(B, 0, device=device)

        if insert_keys:
            site_idx = torch.tensor([k[1] for k in insert_keys], device=device, dtype=torch.long)
            atom_idx = torch.tensor([ALLOWED_ATOMS.index(k[2]) for k in insert_keys], device=device, dtype=torch.long)
            node_site = h_nodes[:, site_idx]
            atom_feat = self.atom_embed(atom_idx)
            atom_feat = atom_feat.unsqueeze(0).expand(B, -1, -1)
            feat = torch.cat([node_site, atom_feat], dim=-1)
            insert_logits = self.insert_head(feat).squeeze(-1)

            feas_list = []
            for key in insert_keys:
                feas = torch.as_tensor(mask[key], device=device)
                if feas.ndim == 0:
                    feas = feas.expand(B)
                feas_list.append(feas)
            feas_tensor = torch.stack(feas_list, dim=1)
            insert_logits = torch.where(
                feas_tensor.bool(), insert_logits, torch.full_like(insert_logits, -torch.inf)
            )
        else:
            insert_logits = torch.empty(B, 0, device=device)

        stop_score = self.stop_head(h_nodes.mean(dim=1)).squeeze(-1)
        stop_feas = mask.get("STOP", torch.ones(B, device=device))
        stop_feas = torch.as_tensor(stop_feas, device=device)
        if stop_feas.ndim == 0:
            stop_feas = stop_feas.expand(B)
        stop_score = torch.where(
            stop_feas.bool(), stop_score, torch.full_like(stop_score, -torch.inf)
        )

        logits = torch.cat([edit_logits, insert_logits, stop_score.unsqueeze(1)], dim=1)
        self._actions = actions
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

