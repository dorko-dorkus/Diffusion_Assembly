import torch
from assembly_diffusion.graph import MoleculeGraph


def test_is_valid_without_rdkit():
    atoms = [6, 6]
    bonds = torch.zeros((2, 2), dtype=torch.int)
    g = MoleculeGraph(atoms, bonds)
    assert g.is_valid() is False
