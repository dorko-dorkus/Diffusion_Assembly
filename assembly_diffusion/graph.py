from rdkit import Chem
from rdkit.Chem import SanitizeMol
import torch

class MoleculeGraph:
    """Simple molecular graph representation."""
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

    def copy(self):
        return MoleculeGraph(self.atoms.copy(), self.bonds.clone())

    def to_rdkit(self):
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
        try:
            mol = self.to_rdkit()
            SanitizeMol(mol)
            return True
        except Exception:
            return False

    def canonical_smiles(self):
        return Chem.MolToSmiles(self.to_rdkit(), canonical=True)
