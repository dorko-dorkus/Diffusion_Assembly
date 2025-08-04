"""Noise schedule utilities for diffusion."""

from __future__ import annotations

import math
from typing import Iterable

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
        graph_batch: Iterable[MoleculeGraph],
        rho_target: float,
        epsilon: float = 1e-3,
        max_steps: int = 100,
    ) -> float:
        """Tune ``beta0`` to match a desired masked bond fraction.

        The expected fraction of masked bonds after ``T`` steps is

        ``f(beta0) = F * (1 - exp(-beta0))``

        where ``F`` is the average fraction of present bonds across the batch.
        ``beta0`` is updated so that ``f(beta0)`` is within ``epsilon`` of
        ``rho_target``.

        Parameters
        ----------
        graph_batch:
            Iterable of :class:`MoleculeGraph` instances.
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

        total_bonds = 0.0
        total_pairs = 0.0
        for g in graph_batch:
            n = len(g.atoms)
            total_pairs += n * (n - 1) / 2
            # Count unique bonds (upper triangular part of adjacency)
            total_bonds += float((g.bonds.triu(1) > 0).sum().item())

        if total_pairs == 0:
            raise ValueError("Graph batch contains no atom pairs")

        F = total_bonds / total_pairs
        if rho_target >= F:
            raise ValueError(
                "Target masked fraction exceeds maximum achievable for batch"
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
