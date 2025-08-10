import pytest
import torch
from rdkit import Chem

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.eval.validity import validity_rate, compare_validity_runs


def _valid_graph():
    mol = Chem.MolFromSmiles("CC")
    return MoleculeGraph.from_rdkit(mol)


def _invalid_graph():
    bonds = torch.tensor([[0, 4], [4, 0]], dtype=torch.int64)
    return MoleculeGraph(["C", "C"], bonds)


def test_validity_rate():
    graphs = [_valid_graph(), _invalid_graph()]
    assert validity_rate(graphs) == pytest.approx(0.5)


def test_compare_validity_runs():
    runs = {
        "baseline": [_invalid_graph()],
        "model": [_valid_graph(), _invalid_graph()],
    }
    results = compare_validity_runs(runs)
    assert results["baseline"] == 0.0
    assert results["model"] == pytest.approx(0.5)
