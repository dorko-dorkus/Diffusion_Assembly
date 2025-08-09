import torch
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.assembly_index import AssemblyIndex


def _path_graph(n: int) -> MoleculeGraph:
    atoms = ["C"] * n
    bonds = torch.zeros((n, n), dtype=torch.int64)
    for i in range(n - 1):
        bonds[i, i + 1] = bonds[i + 1, i] = 1
    return MoleculeGraph(atoms, bonds)


def _cycle_graph(n: int) -> MoleculeGraph:
    g = _path_graph(n)
    g.bonds[0, n - 1] = g.bonds[n - 1, 0] = 1
    return g


def test_acyclic_exact():
    g = _path_graph(4)
    assert g.is_acyclic()
    assert AssemblyIndex.A_star_exact_or_none(g) == g.num_edges()


def test_unicyclic_exact():
    g = _cycle_graph(4)
    mu = g.num_edges() - g.num_nodes() + g.num_connected_components()
    assert mu == 1
    assert AssemblyIndex.A_star_exact_or_none(g) == g.num_edges() + mu


def test_bicyclic_exact():
    g = _cycle_graph(4)
    # add a diagonal to create a second cycle
    g.bonds[1, 3] = g.bonds[3, 1] = 1
    mu = g.num_edges() - g.num_nodes() + g.num_connected_components()
    assert mu == 2
    assert AssemblyIndex.A_star_exact_or_none(g) == g.num_edges() + mu
