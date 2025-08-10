"""Utility functions for QED and synthetic accessibility statistics."""
from __future__ import annotations

from functools import lru_cache
from statistics import median
from typing import List, Tuple, Optional

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
    from rdkit.Chem import QED
    from .sascorer import calculateScore as sa_score
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    QED = None
    sa_score = None


@lru_cache(maxsize=None)
def _qed_from_smiles(smiles: str) -> Optional[float]:
    if Chem is None or QED is None:
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
    if Chem is None or sa_score is None:
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
    for duplicates.
    """
    unique = {s for s in smiles if s}
    qeds = []
    sas = []
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
