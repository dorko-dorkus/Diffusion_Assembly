from __future__ import annotations
from assembly_diffusion.calibrators.strings import StringGrammar
from assembly_diffusion.calibrators.trees import TreeGrammar
from assembly_diffusion.calibrators.sampler import Sampler
import math


def test_Astar_strings():
    assert StringGrammar.A_star("") == 0
    assert StringGrammar.A_star("ABC") == 3


def test_sampler_S_shapes():
    S = Sampler(seed=123)
    df = S.sample_S(L_max=6, n_samp=100, guided=False)
    assert (df["universe"] == "S").all()
    assert df["As_lower"].min() >= 1
    assert df["As_upper"].max() <= 6


def test_Astar_trees():
    gram = TreeGrammar(N_max=6)
    edges = gram.sample_trajectory(6)
    G = gram.to_graph(edges)
    assert gram.A_star(G) == 5


def test_sampler_T_shapes():
    S = Sampler(seed=0)
    df = S.sample_T(N_max=6, n_samp=100, guided=False)
    assert (df["universe"] == "T").all()
    assert (df["As_upper"] == df["As_lower"]).all()
    assert df["As_upper"].min() >= 1


def test_enumerate_strings_closed_form():
    df = StringGrammar.enumerate(L_max=3)
    counts = df.groupby("As_upper").size().to_dict()
    assert counts == {1: 3, 2: 9, 3: 27}
    assert df["frequency"].sum() == 1.0


def test_enumerate_trees_small():
    df = TreeGrammar.enumerate(N_max=5)
    counts = {a: len(g) for a, g in df.groupby("As_upper")}
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 2
    assert counts[4] == 3
    for N in range(2, 6):
        A = N - 1
        sub = df[df["As_upper"] == A]
        assert sub["d_min"].sum() == math.factorial(N - 1)
        assert sub["frequency"].sum() == 1.0
