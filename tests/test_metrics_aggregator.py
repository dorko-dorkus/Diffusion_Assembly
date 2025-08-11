"""Tests for the metrics aggregator utility.

baseline: simple molecule sets provide deterministic metric counts.
validation: ensures validity ratios lie in [0,1], counts sum to the total
    number of molecules and uniqueness/novelty ratios are well behaved. Also
    checks that rows are emitted to the requested CSV file.
"""

import pandas as pd
import pytest
from rdkit import Chem
import torch

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.eval.metrics_aggregator import aggregate_metrics


def _valid(smiles: str) -> MoleculeGraph:
    return MoleculeGraph.from_rdkit(Chem.MolFromSmiles(smiles))


def _invalid() -> MoleculeGraph:
    bonds = torch.tensor([[0, 4], [4, 0]], dtype=torch.int64)
    return MoleculeGraph(["C", "C"], bonds)


@pytest.mark.skipif(Chem is None, reason="RDKit not installed")
def test_aggregate_metrics(tmp_path):
    graphs = [_valid("C"), _invalid(), _valid("O")]
    reference = ["C"]
    out_csv = tmp_path / "metrics.csv"

    rows = aggregate_metrics({"run": graphs}, reference, out_csv)

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert df.shape[0] == 1

    row = rows[0]
    assert 0.0 <= row["validity"] <= 1.0
    assert row["n_total"] == len(graphs)
    assert row["n_valid"] + row["n_invalid"] == row["n_total"]
    assert 0.0 <= row["uniqueness"] <= 1.0
    assert 0.0 <= row["novelty"] <= 1.0
