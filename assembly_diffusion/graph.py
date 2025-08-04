"""Utility functions for representing molecules as graphs."""

import torch

try:
    from rdkit import Chem
    from rdkit.Chem import SanitizeMol
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    SanitizeMol = None

class MoleculeGraph:
    """Simple molecular graph representation."""
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

    def copy(self):
        return MoleculeGraph(self.atoms.copy(), self.bonds.clone())

    def to_rdkit(self):
        """Convert this graph to an RDKit molecule.

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
        atom_map = [mol.AddAtom(Chem.Atom(a)) for a in self.atoms]
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                b = int(self.bonds[i, j])
                if b > 0:
                    bond_type = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE][b - 1]
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
        return mol

    def is_valid(self):
        """Return ``True`` if the molecule can be sanitized by RDKit.

        If RDKit is not available the function returns ``False``.
        """

        if Chem is None or SanitizeMol is None:
            return False
        try:
            mol = self.to_rdkit()
            SanitizeMol(mol)
            return True
        except Exception:
            return False

    def canonical_smiles(self):
        """Return the canonical SMILES string using RDKit.

        Raises
        ------
        ImportError
            If RDKit is not installed.
        """

        if Chem is None:
            raise ImportError("RDKit is required to generate canonical SMILES")
        return Chem.MolToSmiles(self.to_rdkit(), canonical=True)
