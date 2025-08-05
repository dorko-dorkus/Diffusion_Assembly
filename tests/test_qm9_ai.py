import pathlib
import sys
import pytest

# Import torch lazily so the test suite can run without the dependency.
torch = pytest.importorskip("torch")

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion import qm9_ai
from assembly_diffusion import graph as graph_mod


def test_generate_qm9_chon_ai_invokes_mc(tmp_path, monkeypatch):
    graph = MoleculeGraph(["C", "C"], torch.tensor([[0, 1], [1, 0]], dtype=torch.int64))

    # Ensure RDKit conversions fall back to the simple path
    monkeypatch.setattr(qm9_ai, "Chem", None)
    monkeypatch.setattr(graph_mod, "Chem", None)

    # Stub out dataset loader
    monkeypatch.setattr(
        qm9_ai, "load_qm9_chon", lambda max_heavy, data_dir: [graph]
    )

    # Stub surrogate model
    class DummySurrogate:
        def score(self, g):
            assert g is graph
            return 2.0

    monkeypatch.setattr(qm9_ai, "AISurrogate", lambda: DummySurrogate())

    # Track AssemblyMC invocation
    created = {}

    class DummyMC:
        def __init__(self, samples):
            created["samples"] = samples
            created["instance"] = self
            self.calls = 0

        def ai(self, g):
            self.calls += 1
            return 1

    monkeypatch.setattr(qm9_ai, "AssemblyMC", DummyMC)

    out_file = tmp_path / "out.csv"
    qm9_ai.generate_qm9_chon_ai(output_path=str(out_file), samples=7)

    # The dummy AssemblyMC should have been created and invoked exactly once
    assert created["samples"] == 7
    assert created["instance"].calls == 1

    import pandas as pd

    df = pd.read_csv(out_file)
    assert list(df["ai_exact"]) == [1]
    assert list(df["ai_surrogate"]) == [2.0]
    assert list(df["ai_conflict"]) == [1]
