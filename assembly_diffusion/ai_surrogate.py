"""Surrogate scoring model for molecular graphs.

This module provides a light-weight heuristic surrogate model for evaluating
molecular graphs.  The score is constructed from three hand crafted heuristic
components ``C``, ``R`` and ``F`` and mapped into a configurable range
``[S_min, S_max]``.

``C``
    Connection density – the fraction of existing bonds relative to the
    maximum possible number of bonds.
``R``
    Ring heuristic – an approximation of the number of rings normalised by the
    number of atoms.
``F``
    Functional group heuristic – the fraction of atoms that are not carbon.

The :class:`AISurrogate` exposes a :meth:`score` method together with
``delta`` to evaluate the effect of edits, and an optional :meth:`fit` method to
learn the weighting parameters from labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .graph import MoleculeGraph


@dataclass
class AISurrogate:
    """Heuristic surrogate model.

    Parameters
    ----------
    S_min, S_max:
        Bounds for the returned score.  The individual heuristic components are
        assumed to lie in ``[0, 1]`` and are combined linearly using the
        weights ``alpha``, ``beta`` and ``gamma``.
    alpha, beta, gamma:
        Weighting coefficients for the heuristics ``C``, ``R`` and ``F``
        respectively.
    bias:
        Optional bias term used during fitting.
    """

    S_min: float = 0.0
    S_max: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    bias: float = 0.0

    # ------------------------------------------------------------------
    # Heuristic components
    # ------------------------------------------------------------------
    def C(self, graph: MoleculeGraph) -> float:
        """Connection density heuristic.

        Returns the fraction of realised bonds relative to the maximum
        possible number of bonds in the graph.
        """

        bonds = graph.bonds
        n = bonds.shape[0]
        if n <= 1:
            return 0.0
        edges = torch.count_nonzero(torch.triu(bonds, diagonal=1)).item()
        max_edges = n * (n - 1) / 2
        return min(edges / max_edges, 1.0)

    def R(self, graph: MoleculeGraph) -> float:
        """Ring heuristic.

        Approximates the number of rings using ``edges - nodes + components``
        and normalises by the number of atoms.
        """

        bonds = graph.bonds
        n = bonds.shape[0]
        if n == 0:
            return 0.0
        edges = torch.count_nonzero(torch.triu(bonds, diagonal=1)).item()
        # Determine number of connected components via depth first search.
        adj = bonds > 0
        visited = set()
        components = 0
        for start in range(n):
            if start in visited:
                continue
            components += 1
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                neighbours = torch.nonzero(adj[v]).view(-1).tolist()
                for w in neighbours:
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
        rings = edges - n + components
        ratio = rings / n
        return float(max(min(ratio, 1.0), 0.0))

    def F(self, graph: MoleculeGraph) -> float:
        """Functional group heuristic.

        Computes the fraction of atoms that are not carbon.
        """

        atoms = graph.atoms
        if not atoms:
            return 0.0
        hetero = sum(1 for a in atoms if a != "C")
        return hetero / len(atoms)

    # ------------------------------------------------------------------
    # Scoring and edits
    # ------------------------------------------------------------------
    def _scale(self, x: float) -> float:
        """Map ``x`` in ``[0, 1]`` to ``[S_min, S_max]``."""

        x = max(0.0, min(1.0, x))
        return self.S_min + (self.S_max - self.S_min) * x

    def score(self, graph: MoleculeGraph) -> float:
        """Return the heuristic score for ``graph``.

        The score is a weighted combination of the ``C``, ``R`` and ``F``
        heuristics and is guaranteed to lie in ``[S_min, S_max]``.
        """

        c = self.C(graph)
        r = self.R(graph)
        f = self.F(graph)
        total = self.alpha + self.beta + self.gamma
        if total == 0:
            combined = 0.0
        else:
            combined = (self.alpha * c + self.beta * r + self.gamma * f + self.bias) / total
        return self._scale(combined)

    def delta(self, graph: MoleculeGraph, edit: Tuple[int, int, int | None]) -> float:
        """Return the change in score induced by ``edit``.

        Parameters
        ----------
        graph:
            The original graph ``G``.
        edit:
            A tuple ``(i, j, b)`` describing the edit ``e`` applied to the graph
            via :meth:`MoleculeGraph.apply_edit`.

        Returns
        -------
        float
            ``S(τ_e(G)) - S(G)``
        """

        new_graph = graph.apply_edit(*edit)
        return self.score(new_graph) - self.score(graph)

    # ------------------------------------------------------------------
    # Optional fitting of parameters
    # ------------------------------------------------------------------
    def fit(self, graphs: Sequence[MoleculeGraph], labels: Sequence[float]) -> "AISurrogate":
        """Fit weighting parameters to labelled data.

        The method performs a simple linear least squares fit between the
        heuristic features and the provided labels.
        """

        if len(graphs) != len(labels):
            raise ValueError("Number of graphs and labels must match")
        if len(graphs) == 0:
            raise ValueError("No data provided for fitting")

        feats = []
        for g in graphs:
            feats.append([self.C(g), self.R(g), self.F(g), 1.0])
        X = torch.tensor(feats, dtype=torch.double)
        y = torch.tensor(
            [(l - self.S_min) / (self.S_max - self.S_min) for l in labels],
            dtype=torch.double,
        ).unsqueeze(-1)
        # Solve least squares X w = y
        sol = torch.linalg.lstsq(X, y, rcond=None).solution.squeeze()
        self.alpha, self.beta, self.gamma, self.bias = sol.tolist()
        return self
