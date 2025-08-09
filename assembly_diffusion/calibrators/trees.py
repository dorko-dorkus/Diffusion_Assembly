from __future__ import annotations
import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import networkx as nx

@dataclass
class TreeGrammar:
    N_max: int = 10

    def sample_target_N(self, guided: bool = False, gamma: float = 0.0, rng: random.Random | None = None) -> int:
        """Sample target number of nodes.
        Baseline: uniform over {2..N_max}. Guided: softmax tilt toward larger N via exp(gamma * N).
        """
        rng = rng or random
        if not guided or gamma == 0.0:
            return rng.randint(2, self.N_max)
        Ns = list(range(2, self.N_max + 1))
        weights = [pow(2.718281828, gamma * n) for n in Ns]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for n, w in zip(Ns, weights):
            acc += w
            if r <= acc:
                return n
        return self.N_max

    @staticmethod
    def sample_trajectory(N: int, rng: random.Random | None = None) -> List[Tuple[int, int]]:
        """Path-uniform minimal attachment sequences: start from node 0, at step t attach new node t to a uniformly random existing node in {0..t-1}. Returns edge list.
        """
        rng = rng or random
        edges: List[Tuple[int, int]] = []
        for t in range(1, N):
            parent = rng.randint(0, t - 1)
            edges.append((parent, t))
        return edges

    @staticmethod
    def to_graph(edges: List[Tuple[int, int]]) -> nx.Graph:
        G = nx.Graph()
        for u, v in edges:
            G.add_edge(u, v)
        return G

    @staticmethod
    def canonical_id(G: nx.Graph) -> str:
        # Use graph6 canonical string as an isomorphism-invariant id.
        # Relabel nodes to 0..n-1 in some deterministic order first.
        H = nx.convert_node_labels_to_integers(G, label_attribute=None)
        return nx.to_graph6_bytes(H, header=False).decode("ascii")

    @staticmethod
    def A_star(G: nx.Graph) -> int:
        # Minimal steps = N - 1 for a tree with N nodes.
        return G.number_of_nodes() - 1

    @staticmethod
    def D_min_exact(G: nx.Graph, N_limit: int = 9) -> int:
        """Exact count of distinct minimal attachment sequences up to graph isomorphism for small N.
        This enumerates all sequences that attach node t to any existing node in {0..t-1}, then filters by those yielding an isomorphic graph.
        Complexity grows ~ O(N!); restrict to N ≤ 9.
        """
        N = G.number_of_nodes()
        if N > N_limit:
            raise ValueError(f"D_min exact enumeration limited to N ≤ {N_limit}; got N={N}")
        target_id = TreeGrammar.canonical_id(G)
        count = 0
        parent_choices = [list(range(t)) for t in range(1, N)]
        for parents in itertools.product(*parent_choices):
            edges = [(parents[t-1], t) for t in range(1, N)]
            G_seq = TreeGrammar.to_graph(edges)
            if TreeGrammar.canonical_id(G_seq) == target_id:
                count += 1
        return count
