"""Approximate assembly index calculations."""

from typing import Iterable

from .graph import MoleculeGraph

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit.Chem import BRICS
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    BRICS = None


def fragmenter(graph: MoleculeGraph) -> Iterable[str]:
    """Return fragments for ``graph`` using BRICS cuts.

    If RDKit is unavailable a crude fallback concatenates atomic symbols into a
    single fragment.
    """
    if Chem is None or BRICS is None:
        return ["".join(graph.atoms)]
    mol = graph.to_rdkit()
    return list(BRICS.BRICSDecompose(mol))


def approx_AI(graph: MoleculeGraph) -> int:
    """Fast upper bound on assembly depth.

    The value is computed as the sum of the lengths of unique fragments
    obtained from :func:`fragmenter`.
    """
    fragments = fragmenter(graph)
    return sum(len(f) for f in set(fragments))

from .calibrators.strings import StringGrammar
from .calibrators.trees import TreeGrammar


class AssemblyIndex:
    @staticmethod
    def A_star_S(x: str) -> int:
        return StringGrammar.A_star(x)

    @staticmethod
    def D_min_S(x: str) -> int:
        return StringGrammar.D_min(x)

    @staticmethod
    def A_star_T(G) -> int:
        return TreeGrammar.A_star(G)

    @staticmethod
    def D_min_T(G, N_limit: int = 9) -> int:
        return TreeGrammar.D_min_exact(G, N_limit=N_limit)

    @staticmethod
    def A_star_exact_or_none(graph):
        """Return exact A* if ``graph`` is acyclic; otherwise ``None``."""

        if hasattr(graph, "is_acyclic") and graph.is_acyclic():
            return graph.num_edges()
        return None

    @staticmethod
    def A_lower_bound(graph):
        """Lower bound on assembly index via a cycle estimate."""

        E = graph.num_edges()
        V = graph.num_nodes()
        mu = max(0, E - V + graph.num_connected_components())
        return E + mu
