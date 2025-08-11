import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from assembly_diffusion.canonicalize import canonicalize_smiles


def test_roundtrip_and_keying():
    smi = "C1=CC=CC=C1"
    expected_mol = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(expected_mol)
    expected_can = Chem.MolToSmiles(expected_mol, isomericSmiles=True, canonical=True)
    expected_key = inchi.MolToInchiKey(expected_mol)
    can, key = canonicalize_smiles(smi)
    assert can == expected_can
    assert key == expected_key


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        canonicalize_smiles("not a smiles")


def test_tempfile_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    canonicalize_smiles("CC")
    assert list(tmp_path.iterdir()) == []
