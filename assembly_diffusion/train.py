import random
import torch

from .forward import ForwardKernel
from .policy import ReversePolicy
from .mask import FeasibilityMask
from .graph import MoleculeGraph
from .backbone import ATOM_MAP


def teacher_edit(x0: MoleculeGraph, xt: MoleculeGraph):
    """Return the edit transforming ``xt`` toward ``x0`` or ``STOP`` if none."""
    for i in range(len(x0.atoms)):
        for j in range(i + 1, len(x0.atoms)):
            b0 = int(x0.bonds[i, j])
            bt = int(xt.bonds[i, j])
            if b0 != bt:
                return (i, j, b0)
    return "STOP"


def train_epoch(loader, kernel: ForwardKernel, policy: ReversePolicy,
                mask: FeasibilityMask, optimizer, lambda_reg: float = 0.0):
    """Train ``policy`` for one epoch over ``loader``.

    Each batch is processed jointly on the GPU.  For every molecule ``G_0`` a
    timestep ``t`` is sampled, the noisy graph ``G_t`` is generated and all
    graphs are padded to the maximum number of nodes in the batch.  Node and
    edge features are then computed in parallel, logits evaluated against the
    teacher edit ``y`` using cross-entropy with optional entropy
    regularization and gradients are accumulated from the averaged batch loss.
    """

    policy.train()
    metrics = {"loss": 0.0, "accuracy": 0.0, "n": 0}

    for batch in loader:
        optimizer.zero_grad()
        device = next(policy.parameters()).device
        batch_size = len(batch)

        t_list = []
        xt_list = []
        mask_list = []
        target_list = []
        max_n = 0
        for x0 in batch:
            t = random.randint(1, kernel.T)
            xt = kernel.sample_xt(x0, t)
            m = mask.mask_edits(xt)
            target_edit = teacher_edit(x0, xt)

            t_list.append(t)
            xt_list.append(xt)
            mask_list.append(m)
            target_list.append(target_edit)
            max_n = max(max_n, len(xt.atoms))

        atoms = torch.zeros((batch_size, max_n), dtype=torch.long, device=device)
        bonds = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32, device=device)
        for i, g in enumerate(xt_list):
            n = len(g.atoms)
            atom_ids = torch.tensor([ATOM_MAP.get(a, 0) for a in g.atoms], device=device)
            atoms[i, :n] = atom_ids
            bonds[i, :n, :n] = g.bonds.to(device).float()

        t_tensor = torch.tensor(t_list, dtype=torch.long, device=device)
        h_nodes, h_edges = policy.backbone((atoms, bonds), t_tensor)

        batch_loss = torch.tensor(0.0, device=device)
        for i in range(batch_size):
            m = mask_list[i]
            target_edit = target_list[i]
            h_n = h_nodes[i]
            h_e = h_edges[i]

            scores = []
            actions = []
            for key, feasible in m.items():
                if key == "STOP":
                    continue
                ii, jj, b = key
                feat = torch.cat([h_n[ii], h_n[jj], h_e[ii, jj]], dim=-1)
                logit = policy.edit_head(feat)[b]
                if not feasible:
                    logit = torch.tensor(-torch.inf, device=device)
                scores.append(logit)
                actions.append(key)

            stop_score = policy.stop_head(h_n.mean(dim=0)).squeeze()
            if not m.get("STOP", 1):
                stop_score = torch.tensor(-torch.inf, device=device)
            scores.append(stop_score)
            actions.append("STOP")

            logits = torch.stack(scores)
            y = actions.index(target_edit)

            probs = torch.softmax(logits, dim=0)
            ce = -torch.log(probs[y] + 1e-12)
            if lambda_reg > 0:
                entropy = -(probs * torch.log(probs + 1e-12)).sum()
                ce = ce - lambda_reg * entropy

            batch_loss = batch_loss + ce
            pred = probs.argmax().item()
            metrics["accuracy"] += 1 if pred == y else 0
            metrics["loss"] += ce.item()
            metrics["n"] += 1

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()

    metrics["loss"] /= metrics["n"]
    metrics["accuracy"] /= metrics["n"]
    return metrics

