"""Utility functions for QED and synthetic accessibility statistics."""
from __future__ import annotations

from functools import lru_cache
from statistics import median
from typing import List, Optional, Tuple

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
    from rdkit.Chem import QED
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    QED = None

try:  # pragma: no cover - optional SA scorer
    from .sascorer import calculateScore as sa_score
except Exception:  # pragma: no cover - handled at runtime
    sa_score = None

_HAVE_QED = Chem is not None and QED is not None
_HAVE_SA = Chem is not None and sa_score is not None


@lru_cache(maxsize=None)
def _qed_from_smiles(smiles: str) -> Optional[float]:
    """Return the QED score for ``smiles`` or ``None`` if unavailable."""
    if not _HAVE_QED:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        return float(QED.qed(mol))
    except (ValueError, RuntimeError):
        return None


@lru_cache(maxsize=None)
def _sa_from_smiles(smiles: str) -> Optional[float]:
    """Return the synthetic accessibility score or ``None`` if unavailable."""
    if not _HAVE_SA:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        return float(sa_score(mol))
    except (ValueError, RuntimeError):
        return None


def qed_sa_distribution(smiles: List[str]) -> Tuple[float, float, float, float]:
    """Return mean and median QED and SA scores for ``smiles``.

    The computation is cached per SMILES string to avoid recomputing scores
    for duplicates.  If RDKit is unavailable, zeros are returned for all
    statistics.
    """
    if not _HAVE_QED and not _HAVE_SA:
        return 0.0, 0.0, 0.0, 0.0

    unique = {s for s in smiles if s}
    qeds: List[float] = []
    sas: List[float] = []
    for s in unique:
        q = _qed_from_smiles(s)
        if q is not None:
            qeds.append(q)
        a = _sa_from_smiles(s)
        if a is not None:
            sas.append(a)
    qed_mean = sum(qeds) / len(qeds) if qeds else 0.0
    sa_mean = sum(sas) / len(sas) if sas else 0.0
    qed_median = float(median(qeds)) if qeds else 0.0
    sa_median = float(median(sas)) if sas else 0.0
    return qed_mean, qed_median, sa_mean, sa_median
