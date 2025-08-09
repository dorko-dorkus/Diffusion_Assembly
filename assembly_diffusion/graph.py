"""Utility functions for representing molecules as graphs.

This module exposes :class:`MoleculeGraph`, a light‑weight graph data structure
for molecules.  It is intentionally minimal and only depends on ``torch`` and
optionally on :mod:`rdkit` for validation and SMILES generation.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .edit_vocab import Edit


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
    coords:
        Optional ``(N, 3)`` array of 3D coordinates for each atom.  The field
        is present to enable 3D‑aware models but is unused by core graph
        operations.
    """

    atoms: List[str]
    bonds: Tensor
    coords: Optional[Tensor] = None

    def copy(self) -> "MoleculeGraph":
        """Return a deep copy of the molecular graph."""

        return MoleculeGraph(
            self.atoms.copy(),
            self.bonds.clone(),
            None if self.coords is None else self.coords.clone(),
        )

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

        if self.coords is not None:
            new_coords = torch.zeros(
                (n + 1, 3), dtype=self.coords.dtype, device=self.coords.device
            )
            new_coords[:n] = self.coords
            # New atom coordinates initialised at origin; callers may adjust.
            self.coords = new_coords

    # Basic graph statistics -------------------------------------------------
    def num_nodes(self) -> int:
        return len(self.atoms)

    def num_edges(self) -> int:
        # Each bond contributes twice in the symmetric adjacency matrix
        return int((self.bonds > 0).sum().item() // 2)

    def num_connected_components(self) -> int:
        n = len(self.atoms)
        visited = set()
        adj = self.bonds > 0
        comps = 0
        for i in range(n):
            if i in visited:
                continue
            comps += 1
            stack = [i]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                neighbours = torch.nonzero(adj[v]).flatten().tolist()
                stack.extend(nb for nb in neighbours if nb not in visited)
        return comps

    def is_acyclic(self) -> bool:
        return self.num_edges() == self.num_nodes() - self.num_connected_components()

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
        coords: Optional[Tensor] = None
        try:  # pragma: no cover - optional conformer info
            conf = mol.GetConformer()
            coords = torch.tensor(
                [list(conf.GetAtomPosition(i)) for i in range(n)],
                dtype=torch.float32,
            )
        except Exception:
            pass
        return MoleculeGraph(atoms, bonds, coords)

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


def is_edit_legal(edit: "Edit", g: MoleculeGraph) -> bool:
    """Return ``True`` if ``edit`` is legal under the molecular grammar.

    The grammar is defined by :data:`assembly_diffusion.mask.GRAMMAR_SPEC` and
    documented in ``docs/grammar.md``.
    """

    from .mask import GRAMMAR_SPEC  # local import to avoid circular dependency

    if edit.is_stop:
        return True
    if edit.i is None or edit.j is None or edit.b is None:
        return False
    n = len(g.atoms)
    if not (0 <= edit.i < n and 0 <= edit.j < n and edit.i < edit.j):
        return False
    if edit.b not in GRAMMAR_SPEC["bond_orders"]:
        return False

    current = int(g.bonds[edit.i, edit.j])
    deg_i = g.degree(edit.i) - current + edit.b
    deg_j = g.degree(edit.j) - current + edit.b
    max_val = GRAMMAR_SPEC["max_valence"]
    if deg_i > max_val.get(g.atoms[edit.i], 4):
        return False
    if deg_j > max_val.get(g.atoms[edit.j], 4):
        return False
    return True


def apply_edit(edit: "Edit", g: MoleculeGraph) -> MoleculeGraph:
    """Apply ``edit`` to ``g`` if it is legal under the grammar.

    An ``AssertionError`` is raised when the edit violates the grammar; see
    ``docs/grammar.md`` for the specification.
    """

    if not is_edit_legal(edit, g):
        raise AssertionError(f"Illegal edit {edit}")
    if edit.is_stop:
        return g
    return g.apply_edit(edit.i, edit.j, edit.b)
