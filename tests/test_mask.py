import torch
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.mask import FeasibilityMask


def test_mask_edits_respects_is_valid(monkeypatch):
    atoms = [6, 6]
    bonds = torch.zeros((2, 2), dtype=torch.int)
    g = MoleculeGraph(atoms, bonds)
    mask = FeasibilityMask()

    monkeypatch.setattr(MoleculeGraph, "is_valid", lambda self: True)
    res = mask.mask_edits(g)
    assert res['STOP'] == 1
    assert len(res) == 4
    assert all(v == 1 for k, v in res.items())

    monkeypatch.setattr(MoleculeGraph, "is_valid", lambda self: False)
    res = mask.mask_edits(g)
    assert res['STOP'] == 1
    for key, val in res.items():
        if key != 'STOP':
            assert val == 0
