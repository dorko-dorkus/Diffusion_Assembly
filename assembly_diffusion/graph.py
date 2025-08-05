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


# Basic valence caps used for fast feasibility checks.  These constants are
# intentionally kept minimal and mirror those used in :mod:`mask` to avoid a
# circular import.
VALENCE_CAP = {"C": 4, "N": 3, "O": 2, "H": 1}
# Allowed atom types for stochastic insertion moves.
ALLOWED_ATOMS = list(VALENCE_CAP.keys())


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

    # ------------------------------------------------------------------
    # Utility helpers used by forward/reverse diffusion kernels

    def degree(self, i: int) -> int:
        """Return the current degree (total bond order) of atom ``i``."""

        return int(self.bonds[i].sum().item())

    def free_valence(self, i: int) -> int:
        """Return remaining valence capacity for atom ``i``."""

        cap = VALENCE_CAP.get(self.atoms[i], 4)
        return cap - self.degree(i)

    def free_valence_sites(self) -> List[int]:
        """Return indices of atoms that can accept additional bonds."""

        return [i for i in range(len(self.atoms)) if self.free_valence(i) > 0]

    def add_atom(self, atom: str, attach_site: int, bond_order: int = 1):
        """Attach a new ``atom`` to ``attach_site`` with ``bond_order``."""

        n = len(self.atoms)
        self.atoms.append(atom)
        # Expand bond matrix with zeros and connect the new atom
        new_bonds = torch.zeros(
            (n + 1, n + 1), dtype=self.bonds.dtype, device=self.bonds.device
        )
        new_bonds[:n, :n] = self.bonds
        new_bonds[n, attach_site] = new_bonds[attach_site, n] = bond_order
        self.bonds = new_bonds

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
