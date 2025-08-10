import pytest
from assembly_diffusion.eval import metrics


def test_smiles_set_requires_rdkit(monkeypatch):
    monkeypatch.setattr(metrics, "Chem", None)
    with pytest.raises(RuntimeError, match="RDKit required for metric smiles_set"):
        metrics.smiles_set([])


def test_evaluate_requires_rdkit(monkeypatch):
    monkeypatch.setattr(metrics, "Chem", None)
    with pytest.raises(RuntimeError, match="RDKit required for metric evaluate"):
        metrics.Metrics.evaluate([], [])
