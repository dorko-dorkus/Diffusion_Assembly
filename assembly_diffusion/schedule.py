"""Noise schedule utilities for diffusion."""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .graph import MoleculeGraph


class NoiseSchedule:
    """Exponential noise schedule ``alpha(t) = exp(-beta0 * t / T)``.

    Parameters
    ----------
    beta0:
        Base noise rate.
    T:
        Total number of diffusion steps.
    """

    def __init__(self, beta0: float = 0.1, T: int = 10):
        self.beta0 = float(beta0)
        self.T = int(T)

    def alpha(self, t: float) -> float:
        """Return ``alpha(t)`` for time ``t``.

        The function supports ``t`` as an integer or float in ``[0, T]``.
        """

        return math.exp(-self.beta0 * t / self.T)

    def calibrate(
        self,
        graph_batch: Union[
            Iterable[MoleculeGraph],
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ],
        rho_target: float,
        epsilon: float = 1e-3,
        max_steps: int = 100,
    ) -> float:
        """Tune ``beta0`` to match a desired masked bond fraction.

        The function accepts either an iterable of :class:`MoleculeGraph`
        instances or pre-collated tensors as produced by
        :func:`assembly_diffusion.data.collate_graphs`.

        The expected fraction of masked bonds after ``T`` steps is

        ``f(beta0) = F * (1 - exp(-beta0))``

        where ``F`` is the average fraction of present bonds across the batch.
        ``beta0`` is updated so that ``f(beta0)`` is within ``epsilon`` of
        ``rho_target``.

        Parameters
        ----------
        graph_batch:
            Iterable of :class:`MoleculeGraph` objects or collated tensors
            ``(atom_tensor, bond_tensor)`` or just ``bond_tensor``.
        rho_target:
            Desired fraction of bonds that are masked at ``t = T``.
        epsilon:
            Tolerance for the calibration.
        max_steps:
            Maximum number of binary search iterations.

        Returns
        -------
        float
            The calibrated ``beta0`` value.
        """

        # --- Prepare bond tensor -----------------------------------------
        if isinstance(graph_batch, torch.Tensor):
            bond_tensor = graph_batch.float()
        elif (
            isinstance(graph_batch, (tuple, list))
            and len(graph_batch) >= 2
            and isinstance(graph_batch[1], torch.Tensor)
        ):
            bond_tensor = graph_batch[1].float()
        else:
            bond_mats = [g.bonds.float() for g in graph_batch]
            if not bond_mats:
                raise ValueError("Graph batch is empty")
            max_atoms = max(b.size(0) for b in bond_mats)
            padded = [
                F.pad(b, (0, max_atoms - b.size(-1), 0, max_atoms - b.size(-2)))
                for b in bond_mats
            ]
            bond_tensor = pad_sequence(padded, batch_first=True)

        # --- Compute statistics via vectorised tensor ops -----------------
        triu = torch.triu(bond_tensor, diagonal=1)
        total_bonds = (triu > 0).sum().item()

        atom_mask = bond_tensor.abs().sum(dim=-1) > 0
        n_atoms = atom_mask.sum(dim=1).float()
        total_pairs = (n_atoms * (n_atoms - 1) / 2).sum().item()

        if total_pairs == 0:
            raise ValueError("Graph batch contains no atom pairs")

        F = total_bonds / total_pairs
        if rho_target >= F:
            raise ValueError(
                "Target masked fraction exceeds maximum achievable for batch",
            )

        def expected_fraction(beta: float) -> float:
            return F * (1.0 - math.exp(-beta))

        # Binary search for beta within tolerance
        lo, hi = 0.0, 1.0
        while expected_fraction(hi) < rho_target:
            hi *= 2.0
            if hi > 1e6:
                raise RuntimeError("Unable to calibrate beta0; target too high")

        beta = hi
        for _ in range(max_steps):
            beta = 0.5 * (lo + hi)
            rho = expected_fraction(beta)
            if abs(rho - rho_target) < epsilon:
                break
            if rho < rho_target:
                lo = beta
            else:
                hi = beta

        self.beta0 = beta
        return beta


__all__ = ["NoiseSchedule"]
