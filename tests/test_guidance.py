import pathlib
import sys

import pytest

# Import torch lazily so the test suite can run without the dependency.
torch = pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.guidance import AssemblyPrior
from assembly_diffusion.graph import MoleculeGraph


def test_assembly_prior_reweight():
    graph = MoleculeGraph(["C", "C"], torch.zeros((2, 2), dtype=torch.int64))
    logits = torch.zeros(3)
    prior = AssemblyPrior(coeff=0.5, target=1)
    new_logits = prior.reweight(logits.clone(), graph)
    expected = torch.tensor([-0.5, -0.5, 0.0])
    assert torch.allclose(new_logits, expected)
