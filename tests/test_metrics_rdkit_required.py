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


def test_schema():
    """Evaluate returns a dictionary with the expected metric keys."""

    if metrics.Chem is None:
        pytest.skip("RDKit not installed")
    result = metrics.Metrics.evaluate([], [])
    expected = {
        "validity",
        "uniqueness",
        "diversity",
        "novelty",
        "qed_mean",
        "qed_median",
        "sa_mean",
        "sa_median",
    }
    assert expected.issubset(result.keys())
