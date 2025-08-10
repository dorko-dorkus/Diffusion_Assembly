import pytest
from assembly_diffusion.eval import metrics
from assembly_diffusion.graph import MoleculeGraph


@pytest.mark.skipif(metrics.Chem is None, reason="RDKit not installed")
def test_evaluate_with_baselines_returns_metrics():
    from rdkit import Chem

    sample_graph = MoleculeGraph.from_rdkit(Chem.MolFromSmiles("O"))
    baseline_graph = MoleculeGraph.from_rdkit(Chem.MolFromSmiles("C"))

    result = metrics.Metrics.evaluate_with_baselines(
        [sample_graph],
        reference_smiles=[],
        baselines={"control": [baseline_graph]},
    )

    assert "sample" in result
    assert "baselines" in result
    assert "control" in result["baselines"]
    assert "validity" in result["sample"]
    assert "validity" in result["baselines"]["control"]
