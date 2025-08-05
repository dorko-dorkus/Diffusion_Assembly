"""Monte Carlo assembly index calculation."""
from __future__ import annotations

import random
from dataclasses import dataclass
import torch

from .graph import MoleculeGraph


@dataclass
class AssemblyMC:
    """Estimate exact assembly index via Monte Carlo search.

    Parameters
    ----------
    samples:
        Number of random assembly trials to perform.
    """

    samples: int = 100

    def ai(self, graph: MoleculeGraph) -> int:
        """Return the shortest disassembly sequence length found.

        The method performs ``samples`` random bond deletion sequences and
        returns the minimal number of deletions required to fragment the
        molecule into isolated atoms.
        """

        min_steps = float("inf")
        for _ in range(self.samples):
            g = graph.copy()
            steps = 0
            edges = torch.nonzero(torch.triu(g.bonds, diagonal=1)).tolist()
            while edges:
                i, j = random.choice(edges)
                g = g.apply_edit(i, j, None)
                steps += 1
                edges = torch.nonzero(torch.triu(g.bonds, diagonal=1)).tolist()
            if steps < min_steps:
                min_steps = steps
        if min_steps == float("inf"):
            return 0
        return int(min_steps)
