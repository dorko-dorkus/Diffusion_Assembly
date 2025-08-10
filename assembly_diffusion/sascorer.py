"""Synthetic accessibility scoring utilities.

This module provides a tiny wrapper around RDKit to compute a rough
synthetic accessibility (SA) score for a molecule.  RDKit is an optional
dependency; when it is not installed the :func:`calculateScore` function will
raise :class:`ImportError`.
"""

from __future__ import annotations

from math import log

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None  # type: ignore
    rdMolDescriptors = None  # type: ignore


def calculateScore(mol: "Chem.Mol") -> float:
    """Return a simple synthetic accessibility score for ``mol``.

    The score is based on a few heuristic RDKit descriptors and is **not** the
    original `sascorer` implementation distributed with RDKit.  It is provided
    as a light‑weight proxy that avoids shipping additional data files.  When
    RDKit is unavailable an :class:`ImportError` is raised.
    """

    if Chem is None or rdMolDescriptors is None:  # pragma: no cover - runtime check
        raise ImportError(
            "RDKit is required for synthetic accessibility scoring"
        )

    # Use accessible surface area as a cheap proxy for synthetic complexity.
    asa = rdMolDescriptors.CalcLabuteASA(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = mol.GetRingInfo().NumRings()

    # Combine features into a rough 1‑10 score similar in scale to the
    # reference implementation.  The exact formula is heuristic and merely
    # intended to provide a monotonically increasing difficulty measure.
    score = (asa / 10.0) + (rings * 0.5) + log(heavy + 1.0)
    return float(score)


__all__ = ["calculateScore"]

