"""Validation utilities for molecular graphs."""

from __future__ import annotations

from ..graph import MoleculeGraph

try:  # pragma: no cover - RDKit optional
    from rdkit.Chem.rdchem import MolSanitizeException
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover - handled at runtime
    MolSanitizeException = RuntimeError
    _HAS_RDKIT = False


def sanitize_or_none(graph: MoleculeGraph):
    """Return a sanitized RDKit molecule or ``None``.

    Parameters
    ----------
    graph:
        The molecular graph to sanitize.

    Returns
    -------
    rdkit.Chem.Mol | None
        Sanitized molecule if successful, otherwise ``None``.
    """

    if not _HAS_RDKIT:
        return None

    try:
        return graph.to_rdkit()
    except (ValueError, RuntimeError, ImportError, MolSanitizeException):
        return None


def is_valid(graph: MoleculeGraph) -> bool:
    """Return ``True`` if ``graph`` can be sanitized by RDKit."""

    return sanitize_or_none(graph) is not None
