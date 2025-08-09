from __future__ import annotations
import random
from typing import Dict, Any, Iterable
import pandas as pd

from .strings import StringGrammar
from .trees import TreeGrammar

class Sampler:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def sample_S(self, L_max: int, n_samp: int, guided: bool = False, gamma: float = 0.0) -> pd.DataFrame:
        gram = StringGrammar(L_max=L_max)
        rows = []
        for _ in range(n_samp):
            L = gram.sample_target_length(guided=guided, gamma=gamma, rng=self.rng)
            traj = gram.sample_trajectory(L, rng=self.rng)
            x = gram.to_object(traj)
            cid = gram.canonical_id(x)
            A = gram.A_star(x)
            rows.append({
                "id": cid,
                "universe": "S",
                "grammar": f"S_v0_L{L_max}",
                "As_lower": A,
                "As_upper": A,
                "validity": 1,
                "frequency": 1.0,
                "d_min": StringGrammar.D_min(x)
            })
        return pd.DataFrame(rows)

    def sample_T(self, N_max: int, n_samp: int, guided: bool = False, gamma: float = 0.0, dmin_exact: bool = False) -> pd.DataFrame:
        gram = TreeGrammar(N_max=N_max)
        rows = []
        for _ in range(n_samp):
            N = gram.sample_target_N(guided=guided, gamma=gamma, rng=self.rng)
            edges = gram.sample_trajectory(N, rng=self.rng)
            G = gram.to_graph(edges)
            cid = gram.canonical_id(G)
            A = gram.A_star(G)
            if dmin_exact and N <= 9:
                dmin = gram.D_min_exact(G, N_limit=9)
            else:
                dmin = None
            rows.append({
                "id": cid,
                "universe": "T",
                "grammar": f"T_v0_N{N_max}",
                "As_lower": A,
                "As_upper": A,
                "validity": 1,
                "frequency": 1.0,
                "d_min": dmin
            })
        return pd.DataFrame(rows)
