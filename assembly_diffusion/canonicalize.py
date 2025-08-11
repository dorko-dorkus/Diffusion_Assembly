"""Utilities for canonicalising SMILES strings."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple, TYPE_CHECKING, Any

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.rdchem import MolSanitizeException
    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None  # type: ignore[assignment]
    inchi = None  # type: ignore[assignment]
    MolSanitizeException = RuntimeError
    _HAVE_RDKIT = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit.Chem.rdchem import Mol
else:  # pragma: no cover - typing fallback
    Mol = Any  # type: ignore[misc,assignment]

__all__ = ["smiles_to_mol", "mol_to_smiles", "canonicalize_smiles"]


def _require_rdkit() -> None:
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for canonicalisation; install via "
            "'conda install -c conda-forge rdkit'."
        )


def smiles_to_mol(smiles: str) -> Mol:
    """Convert a SMILES string to an RDKit Mol.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError, MolSanitizeException) as e:  # pragma: no cover - defensive
        raise ValueError(f"Invalid SMILES: {smiles}") from e
    return mol


def mol_to_smiles(mol: Mol) -> str:
    """Return the canonical SMILES representation of a molecule."""
    _require_rdkit()
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def canonicalize_smiles(smiles: str) -> Tuple[str, str]:
    """Return canonical SMILES and InChIKey for a SMILES string.

    A temporary file is created during processing and removed afterwards to
    ensure no artefacts are left on disk.
    """
    mol = smiles_to_mol(smiles)
    canonical = mol_to_smiles(mol)
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        tmp.write(canonical)
        tmp.flush()
        key = inchi.MolToInchiKey(mol)
    finally:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
    return canonical, key
