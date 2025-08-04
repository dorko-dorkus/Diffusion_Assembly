"""Utility functions for representing molecules as graphs.

This module exposes :class:`MoleculeGraph`, a light‑weight graph data structure
for molecules.  It is intentionally minimal and only depends on ``torch`` and
optionally on :mod:`rdkit` for validation and SMILES generation.
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

try:
    from rdkit import Chem
    from rdkit.Chem import SanitizeMol
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    SanitizeMol = None


@dataclass
class MoleculeGraph:
    """Simple molecular graph representation.

    Attributes
    ----------
    atoms:
        A list of atomic symbols for each atom in the molecule.
    bonds:
        A square adjacency matrix with integer bond orders.  The matrix is
        symmetric and ``bonds[i, j] > 0`` indicates a bond.
    """

    atoms: List[str]
    bonds: Tensor

    def copy(self) -> "MoleculeGraph":
        """Return a deep copy of the molecular graph."""

        return MoleculeGraph(self.atoms.copy(), self.bonds.clone())

    @staticmethod
    def from_rdkit(mol: "Chem.Mol") -> "MoleculeGraph":
        """Construct a :class:`MoleculeGraph` from an RDKit molecule."""

        if Chem is None:
            raise ImportError(
                "RDKit is required for converting an RDKit Mol to MoleculeGraph"
            )
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        n = len(atoms)
        bonds = torch.zeros((n, n), dtype=torch.int64)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = int(bond.GetBondTypeAsDouble())
            bonds[i, j] = bonds[j, i] = order
        return MoleculeGraph(atoms, bonds)

    def to_rdkit(self) -> "Chem.Mol":
        """Convert this graph to an RDKit molecule.

        Returns
        -------
        rdkit.Chem.Mol
            The RDKit molecule created from the graph.

        Raises
        ------
        ImportError
            If RDKit is not installed.
        """

        if Chem is None:
            raise ImportError(
                "RDKit is required for converting MoleculeGraph to an RDKit Mol"
            )
        mol = Chem.RWMol()
        # Add all atoms to the RDKit molecule and keep a mapping from our index
        # to the RDKit atom index.
        atom_map = [mol.AddAtom(Chem.Atom(a)) for a in self.atoms]
        bond_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                b = int(self.bonds[i, j])
                if b > 0:
                    bond_type = bond_map.get(b)
                    if bond_type is None:
                        raise ValueError(f"Unsupported bond order: {b}")
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
        mol = mol.GetMol()
        SanitizeMol(mol)
        return mol

    def is_valid(self) -> bool:
        """Return ``True`` if the molecule can be sanitized by RDKit.

        The method returns ``False`` when RDKit is not available.
        """

        if Chem is None:
            return False
        try:
            self.to_rdkit()
            return True
        except Exception:
            return False

    def apply_edit(self, i: int, j: int, b: int | None) -> "MoleculeGraph":
        """Return a new graph with bond ``(i, j)`` set to ``b``."""

        new = self.copy()
        if b is None or b == 0:
            new.bonds[i, j] = 0
            new.bonds[j, i] = 0
        else:
            new.bonds[i, j] = b
            new.bonds[j, i] = b
        return new

    def canonical_smiles(self) -> str:
        """Return the canonical SMILES string using RDKit.

        Raises
        ------
        ImportError
            If RDKit is not installed.
        """

        if Chem is None:
            raise ImportError("RDKit is required to generate canonical SMILES")
        return Chem.MolToSmiles(self.to_rdkit(), canonical=True)
