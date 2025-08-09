import pytest

torch = pytest.importorskip("torch")

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.guidance import AssemblyPrior
from assembly_diffusion.se3_guidance import DummySE3Flow, GeometryPrior, SE3Guidance


def make_graph():
    atoms = ["C", "C"]
    bonds = torch.zeros((2, 2), dtype=torch.int64)
    coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    return MoleculeGraph(atoms, bonds, coords)


def test_se3_guidance_ablation():
    graph = make_graph()
    logits = torch.zeros(3)  # two actions + STOP
    mask = torch.ones(2, dtype=torch.bool)

    flow = DummySE3Flow()
    geom = GeometryPrior()
    prior = AssemblyPrior(coeff=0.1)
    base = SE3Guidance(flow)
    g_only = SE3Guidance(flow, geometry_prior=geom)
    a_only = SE3Guidance(flow, assembly_prior=prior)
    both = SE3Guidance(flow, assembly_prior=prior, geometry_prior=geom)

    d_base = base(logits, graph, 0, mask)
    d_g = g_only(logits, graph, 0, mask) - d_base
    d_a = a_only(logits, graph, 0, mask) - d_base
    d_b = both(logits, graph, 0, mask)

    # Combined guidance should equal the base flow adjustment plus individual
    # geometry and assembly contributions.
    assert torch.allclose(d_b, d_base + d_g + d_a)
