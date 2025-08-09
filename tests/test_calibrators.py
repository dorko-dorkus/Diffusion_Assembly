from __future__ import annotations
import pandas as pd
from assembly_diffusion.calibrators.strings import StringGrammar
from assembly_diffusion.calibrators.trees import TreeGrammar
from assembly_diffusion.calibrators.sampler import Sampler


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
