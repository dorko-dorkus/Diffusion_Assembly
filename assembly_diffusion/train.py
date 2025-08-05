import os
import random
import logging
import torch

from .forward import ForwardKernel
from .policy import ReversePolicy
from .mask import FeasibilityMask
from .graph import MoleculeGraph
from .backbone import ATOM_TYPES


logger = logging.getLogger(__name__)


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
                mask: FeasibilityMask, optimizer, lambda_reg: float = 0.0,
                *, epoch: int = 0, writer=None, ckpt_interval: int = 500):
    """Train ``policy`` for one epoch over ``loader``.

    The ``loader`` is expected to yield tuples ``(atom_tensor, bond_tensor)``
    where each element contains a batch of atom identifiers and bond adjacency
    matrices.  For every molecule ``G_0`` in the batch a timestep ``t`` is
    sampled, the noisy graph ``G_t`` is generated, logits ``z`` are computed
    and compared against the teacher edit ``y`` using cross-entropy with
    optional entropy regularization.  Metrics are accumulated and model
    parameters are updated using the averaged batch loss.
    """

    policy.train()
    metrics = {"loss": 0.0, "accuracy": 0.0, "n": 0}
    global_step = epoch * len(loader)
    device = next(policy.parameters()).device

    if ckpt_interval:
        os.makedirs("checkpoints", exist_ok=True)

    for i, (atom_tensor, bond_tensor) in enumerate(loader):
        optimizer.zero_grad()
        bond_tensor = bond_tensor.to(device, non_blocking=True)

        # --- Construct batched MoleculeGraph ------------------------------
        atom_lists = []
        for atoms in atom_tensor:
            if isinstance(atoms, torch.Tensor):
                atom_ids = [int(a) for a in atoms.tolist() if int(a) >= 0]
                atom_lists.append([ATOM_TYPES[i] for i in atom_ids])
            else:
                atom_lists.append(list(atoms))
        x0 = MoleculeGraph(atom_lists, bond_tensor)

        B = len(atom_lists)
        t = torch.randint(1, kernel.T + 1, (B,), device=device)
        xt = kernel.sample_xt(x0, t)
        m = mask.mask_edits(xt)
        logits = policy.logits(xt, t, m)

        # --- Compute teacher edits and targets ----------------------------
        x0_graphs = [MoleculeGraph(atom_lists[i], x0.bonds[i]) for i in range(B)]
        xt_graphs = [MoleculeGraph(atom_lists[i], xt.bonds[i]) for i in range(B)]
        targets = [policy._actions.index(teacher_edit(g0, gt))
                   for g0, gt in zip(x0_graphs, xt_graphs)]
        y = torch.tensor(targets, device=device)

        # --- Loss and metrics ---------------------------------------------
        probs = torch.softmax(logits, dim=1)
        ce = -torch.log(probs[torch.arange(B, device=device), y] + 1e-12)
        if lambda_reg > 0:
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            ce = ce - lambda_reg * entropy

        batch_loss = ce.mean()
        batch_loss.backward()
        optimizer.step()

        preds = probs.argmax(dim=1)
        batch_acc = (preds == y).float().mean().item()

        if writer is not None:
            writer.add_scalar("loss/train", batch_loss.item(), global_step + i)
            writer.add_scalar("accuracy/train", batch_acc, global_step + i)

        logger.debug(
            "Epoch %d Step %d/%d - loss: %.4f acc: %.4f",
            epoch,
            i + 1,
            len(loader),
            batch_loss.item(),
            batch_acc,
        )

        if ckpt_interval and (global_step + i) % ckpt_interval == 0:
            path = f"checkpoints/epoch{epoch}_step{global_step + i}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)

        metrics["accuracy"] += (preds == y).sum().item()
        metrics["loss"] += ce.sum().item()
        metrics["n"] += B

    metrics["loss"] /= metrics["n"]
    metrics["accuracy"] /= metrics["n"]
    logger.info(
        "Epoch %d complete - loss: %.3f accuracy: %.3f",
        epoch,
        metrics["loss"],
        metrics["accuracy"],
    )
    return metrics

