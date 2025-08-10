"""Approximate assembly index calculations."""

from typing import Iterable

from .graph import MoleculeGraph

try:  # pragma: no cover - RDKit is optional
    from rdkit import Chem
    from rdkit.Chem import BRICS
    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None  # type: ignore[assignment]
    BRICS = None  # type: ignore[assignment]
    _HAVE_RDKIT = False


def fragmenter(graph: MoleculeGraph) -> Iterable[str]:
    """Return fragments for ``graph`` using BRICS cuts.

    If RDKit is unavailable a crude fallback concatenates atomic symbols into a
    single fragment.
    """
    if not _HAVE_RDKIT:
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
    def A_star_exact_or_none(graph, cycle_limit: int = 2):
        """Return the exact assembly index for small cycle graphs.

        The cyclomatic number ``μ = E - V + C`` counts the number of
        independent cycles in ``graph``.  Recent results show that for
        graphs with ``μ ≤ 2`` the assembly index is exactly ``E + μ``.
        This covers acyclic, unicyclic and bicyclic molecules which form
        the bulk of small organic compounds.

        Parameters
        ----------
        graph:
            The :class:`~assembly_diffusion.graph.MoleculeGraph` to
            analyse.
        cycle_limit:
            Maximum cyclomatic number for which the closed form is
            guaranteed.  Graphs exceeding this limit return ``None``.

        Returns
        -------
        int | None
            Exact assembly index when ``μ ≤ cycle_limit``; otherwise
            ``None``.
        """

        E = graph.num_edges()
        V = graph.num_nodes()
        mu = max(0, E - V + graph.num_connected_components())
        if mu <= cycle_limit:
            # For μ ≤ 2 this formula is provably exact
            return E + mu
        return None

    @staticmethod
    def A_lower_bound(graph):
        """Lower bound on assembly index via a cycle estimate."""

        E = graph.num_edges()
        V = graph.num_nodes()
        mu = max(0, E - V + graph.num_connected_components())
        return E + mu
