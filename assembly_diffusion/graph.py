"""Utility functions for representing molecules as graphs.

This module exposes :class:`MoleculeGraph`, a light‑weight graph data structure
for molecules.  It is intentionally minimal and only depends on ``torch`` and
optionally on :mod:`rdkit` for validation and SMILES generation.
"""

from dataclasses import dataclass
from typing import List

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
        A list of atomic numbers for each atom in the molecule.
    bonds:
        A square adjacency matrix with integer bond orders.  The matrix is
        symmetric and ``bonds[i, j] > 0`` indicates a bond.
    """

    atoms: List[int]
    bonds: Tensor

    def copy(self) -> "MoleculeGraph":
        """Return a deep copy of the molecular graph."""

        return MoleculeGraph(self.atoms.copy(), self.bonds.clone())

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
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                b = int(self.bonds[i, j])
                if b > 0:
                    # Map the bond order to the appropriate RDKit bond type.
                    bond_type = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE][b - 1]
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
        return mol

    def is_valid(self) -> bool:
        """Return ``True`` if the molecule can be sanitized by RDKit.

        The method returns ``False`` when RDKit is not available.
        """

        if Chem is None or SanitizeMol is None:
            return False
        try:
            mol = self.to_rdkit()
            SanitizeMol(mol)
            return True
        except Exception:
            return False

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
