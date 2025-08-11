"""Utilities for canonicalising SMILES strings."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.rdchem import Mol, MolSanitizeException

__all__ = ["smiles_to_mol", "mol_to_smiles", "canonicalize_smiles"]


def smiles_to_mol(smiles: str) -> Mol:
    """Convert a SMILES string to an RDKit Mol.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
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
