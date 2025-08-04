import random
import torch

from .forward import ForwardKernel
from .policy import ReversePolicy
from .mask import FeasibilityMask
from .graph import MoleculeGraph


def teacher_edit(x0: MoleculeGraph, xt: MoleculeGraph):
    """Return a random edit transforming ``xt`` toward ``x0`` or ``STOP`` if none."""
    diff = []
    n = len(x0.atoms)
    for i in range(n):
        for j in range(i + 1, n):
            b0 = int(x0.bonds[i, j])
            bt = int(xt.bonds[i, j])
            if b0 != bt:
                diff.append((i, j, b0))
    if not diff:
        return "STOP"
    return random.choice(diff)


def train_epoch(loader, kernel: ForwardKernel, policy: ReversePolicy,
                mask: FeasibilityMask, optimizer, lambda_reg: float = 0.0):
    """Train ``policy`` for one epoch over ``loader``.

    Each batch is processed element-wise: for every molecule ``G_0`` a timestep
    ``t`` is sampled, the noisy graph ``G_t`` is generated, logits ``z`` are
    computed and compared against the teacher edit ``y`` using cross-entropy
    with optional entropy regularization.  Metrics are accumulated and model
    parameters are updated using the averaged batch loss.
    """

    policy.train()
    metrics = {"loss": 0.0, "accuracy": 0.0, "n": 0}

    for batch in loader:
        optimizer.zero_grad()
        device = next(policy.parameters()).device
        batch_loss = torch.tensor(0.0, device=device)

        for x0 in batch:
            t = random.randint(1, kernel.T)
            xt = kernel.sample_xt(x0, t)
            m = mask.mask_edits(xt)
            target_edit = teacher_edit(x0, xt)
            logits = policy.logits(xt, t, m)
            y = policy._actions.index(target_edit)

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

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()

    metrics["loss"] /= metrics["n"]
    metrics["accuracy"] /= metrics["n"]
    return metrics

