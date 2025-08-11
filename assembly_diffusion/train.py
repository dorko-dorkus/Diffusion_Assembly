"""Training utilities for the diffusion policy network.

baseline: standard teacher-forced cross-entropy training without additional
    heuristics or guidance.
data_sources: batched atom and bond tensors from datasets such as the
    QM9-CHON subset provided by the data loader.
method: sample a timestep, generate ``G_t`` via :class:`~assembly_diffusion.forward.ForwardKernel`,
    compute logits from :class:`~assembly_diffusion.policy.ReversePolicy` under
    feasibility masks and optimise using cross-entropy with optional entropy
    regularisation.
metrics: average loss and accuracy over the epoch.
objective: learn a policy that predicts edits transforming noisy graphs back
    toward ground-truth molecules.
repro: deterministic seeds, saved checkpoints and committed configs enable
    reproducible training runs.
validation: data can be partitioned into train/validation/test splits for model
    selection or used in cross-validation. Early stopping monitors a validation
    metric with configurable patience to avoid overfitting, and optional
    :class:`~assembly_diffusion.monitor.RunMonitor` checkpoints verify correct
    behaviour of the training loop.
"""

import os
import random
import time
from typing import Union

import torch
from torch.utils.data import DataLoader, random_split

from .forward import ForwardKernel
from .policy import ReversePolicy
from .mask import FeasibilityMask
from .graph import MoleculeGraph
from .backbone import ATOM_TYPES
from .monitor import RunMonitor
from .data import collate_graphs
from .logging_config import get_logger


logger = get_logger(__name__)


def teacher_edit(x0: MoleculeGraph, xt: MoleculeGraph) -> Union[str, tuple[int, int, int]]:
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


def random_baseline_accuracy(
    loader,
    kernel: ForwardKernel,
    policy: ReversePolicy,
) -> float:
    """Return accuracy of a random policy for comparison.

    The function mirrors the data handling in :func:`train_epoch` but samples
    random edits instead of using ``policy.logits``.  It provides a simple
    control baseline to contextualise training accuracy.
    """

    policy.eval()
    device = next(policy.parameters()).device
    n_actions = len(policy._actions)
    correct = 0
    total = 0

    for atom_tensor, bond_tensor in loader:
        bond_tensor = bond_tensor.to(device, non_blocking=True)

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

        x0_graphs = [MoleculeGraph(atom_lists[i], x0.bonds[i]) for i in range(B)]
        xt_graphs = [MoleculeGraph(atom_lists[i], xt.bonds[i]) for i in range(B)]
        targets = [policy._actions.index(teacher_edit(g0, gt))
                   for g0, gt in zip(x0_graphs, xt_graphs)]
        y = torch.tensor(targets, device=device)

        rand = torch.randint(0, n_actions, (B,), device=device)
        correct += (rand == y).sum().item()
        total += B

    return correct / total if total else 0.0


def split_dataset(
    dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    *,
    seed: int = 0,
):
    """Return train, validation and test subsets from ``dataset``.

    The split is deterministic given ``seed``.  Validation data are used for
    model selection and to drive early stopping while the test subset is held
    out for final reporting.  Ratios must sum to one; to perform cross-
    validation, call this function repeatedly with different ``seed`` values and
    aggregate the results.
    """

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def evaluate_epoch(
    loader,
    kernel: ForwardKernel,
    policy: ReversePolicy,
    mask: FeasibilityMask,
) -> dict[str, float]:
    """Return loss and accuracy over ``loader`` without gradient updates."""

    policy.eval()
    metrics = {"loss": 0.0, "accuracy": 0.0, "n": 0}
    device = next(policy.parameters()).device

    with torch.no_grad():
        for atom_tensor, bond_tensor in loader:
            bond_tensor = bond_tensor.to(device, non_blocking=True)

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

            x0_graphs = [MoleculeGraph(atom_lists[i], x0.bonds[i]) for i in range(B)]
            xt_graphs = [MoleculeGraph(atom_lists[i], xt.bonds[i]) for i in range(B)]
            targets = [policy._actions.index(teacher_edit(g0, gt))
                       for g0, gt in zip(x0_graphs, xt_graphs)]
            y = torch.tensor(targets, device=device)

            probs = torch.softmax(logits, dim=1)
            ce = -torch.log(probs[torch.arange(B, device=device), y] + 1e-12)
            preds = probs.argmax(dim=1)

            metrics["accuracy"] += (preds == y).sum().item()
            metrics["loss"] += ce.sum().item()
            metrics["n"] += B

    metrics["loss"] /= metrics["n"]
    metrics["accuracy"] /= metrics["n"]
    return metrics


def train_epoch(
    loader,
    kernel: ForwardKernel,
    policy: ReversePolicy,
    mask: FeasibilityMask,
    optimizer,
    lambda_reg: float = 0.0,
    *,
    epoch: int = 0,
    writer=None,
    ckpt_interval: int | None = 500,
    monitor: RunMonitor | None = None,
) -> dict[str, float]:
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

    total_steps = len(loader)
    step_in_epoch = 0
    last_poke_check = time.time()

    if ckpt_interval or monitor:
        os.makedirs("checkpoints", exist_ok=True)

    for i, (atom_tensor, bond_tensor) in enumerate(loader):
        step_in_epoch += 1
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

        # Periodic non-blocking status
        if monitor and (step_in_epoch % 10 == 0):
            monitor.tick(step=step_in_epoch, total=total_steps)
        if monitor and (step_in_epoch % 50 == 0):
            try:
                monitor.scalar(
                    "loss/train", float(batch_loss.detach().cpu()), step=step_in_epoch
                )
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("monitor.scalar failed: %s", e)
        if monitor and (step_in_epoch % 100 == 0):
            try:
                smi = xt_graphs[0].canonical_smiles()
                monitor.sample_smiles([smi], step=step_in_epoch)
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("monitor.sample_smiles failed: %s", e)

        if monitor and (time.time() - last_poke_check) > 5:
            ckpt_req, dump_req = monitor.poll()
            last_poke_check = time.time()
            if dump_req:
                monitor.tick(step=step_in_epoch, total=total_steps)
            if ckpt_req:
                tmp = f"checkpoints/epoch{epoch}_step{step_in_epoch}.tmp"
                final = f"checkpoints/epoch{epoch}_step{step_in_epoch}.pt"
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                    },
                    tmp,
                )
                os.replace(tmp, final)
                monitor.set_checkpoint(final)

        # Console status line
        if (step_in_epoch % 10) == 0:
            eta_s = "?"
            logger.info(
                "[epoch %03d] step %06d/%d loss=%.4f ETA~%ss",
                epoch,
                step_in_epoch,
                total_steps,
                float(batch_loss),
                eta_s,
            )

        if ckpt_interval and (step_in_epoch % ckpt_interval == 0):
            tmp = f"checkpoints/epoch{epoch}_step{step_in_epoch}.tmp"
            final = f"checkpoints/epoch{epoch}_step{step_in_epoch}.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                },
                tmp,
            )
            os.replace(tmp, final)
            if monitor:
                monitor.set_checkpoint(final)

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


def train_model(
    dataset,
    kernel: ForwardKernel,
    policy: ReversePolicy,
    mask: FeasibilityMask,
    optimizer,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    patience: int = 5,
    seed: int = 0,
    writer=None,
    ckpt_interval: int | None = 500,
    monitor: RunMonitor | None = None,
) -> dict[str, dict[str, float] | list[float]]:
    """Train ``policy`` with validation-based early stopping.

    ``dataset`` is split into train/validation/test subsets using
    :func:`split_dataset`.  Model selection uses the validation loss and
    training terminates when this metric fails to improve for ``patience``
    consecutive epochs.  The returned dictionary contains metrics for the
    train, validation and test loaders alongside the validation loss history.
    """

    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=1 - val_ratio - test_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_graphs,
            pin_memory=torch.cuda.is_available(),
            num_workers=min(4, os.cpu_count() or 0),
        )

    train_loader = make_loader(train_set, True)
    val_loader = make_loader(val_set, False)
    test_loader = make_loader(test_set, False)

    best_val = float("inf")
    no_improve = 0
    val_history: list[float] = []
    train_metrics = {}
    val_metrics = {}
    for epoch in range(epochs):
        train_metrics = train_epoch(
            train_loader,
            kernel,
            policy,
            mask,
            optimizer,
            epoch=epoch,
            writer=writer,
            ckpt_interval=ckpt_interval,
            monitor=monitor,
        )
        val_metrics = evaluate_epoch(val_loader, kernel, policy, mask)
        val_history.append(val_metrics["loss"])
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    test_metrics = evaluate_epoch(test_loader, kernel, policy, mask)
    return {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "val_history": val_history,
    }
