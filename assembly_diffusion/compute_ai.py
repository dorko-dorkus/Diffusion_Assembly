from __future__ import annotations

"""Facade for computing assembly index bounds.

This module exposes :func:`compute_ai` which evaluates the assembly index for
``MoleculeGraph`` instances using either a fast surrogate or the external
AssemblyMC solver.  The behaviour is controlled via ``method`` which mirrors
``ai.method`` from the run configuration.

* ``method="surrogate"`` – return a pair ``(As_lower, As_upper)`` computed via
  inexpensive graph heuristics.
* ``method="assemblymc"`` – call the AssemblyMC binary to obtain the exact
  assembly index ``A_star``.  The lower and upper bounds are both set to this
  value.

When ``method="assemblymc"`` but the required ``ASSEMBLYMC_BIN`` environment
variable is not defined, the function raises :class:`AssemblyMCError` unless
``allow_fallback`` is set.  In the latter case the surrogate bounds are
returned instead.
"""

from typing import Tuple

from .graph import MoleculeGraph
from .assembly_index import AssemblyIndex, approx_AI


def _surrogate_bounds(graph: MoleculeGraph) -> Tuple[int, int]:
    """Return heuristic lower and upper bounds for ``graph``.

    The lower bound is based on a cycle count and the upper bound uses the
    fragment based approximation.  The upper bound is never smaller than the
    lower bound.
    """

    lower = AssemblyIndex.A_lower_bound(graph)
    upper = max(lower, approx_AI(graph))
    return int(lower), int(upper)


def compute_ai(
    graph: MoleculeGraph,
    method: str = "surrogate",
    allow_fallback: bool = False,
) -> Tuple[int, int]:
    """Compute assembly index bounds for ``graph``.

    Parameters
    ----------
    graph:
        Molecule graph to evaluate.
    method:
        ``"surrogate"`` or ``"assemblymc"``.
    allow_fallback:
        When ``True`` and ``method="assemblymc"`` but the AssemblyMC binary is
        unavailable, the surrogate bounds are returned instead of raising.

    Returns
    -------
    Tuple[int, int]
        ``(As_lower, As_upper)`` bounds on the assembly index.
    """

    if method == "surrogate":
        return _surrogate_bounds(graph)

    if method == "assemblymc":
        try:
            from rdkit import Chem  # type: ignore
            from .extern.assemblymc import a_star_and_dmin, AssemblyMCError

            mol = graph.to_rdkit()
            smiles = Chem.MolToSmiles(mol, canonical=True)
            a_star, _, _ = a_star_and_dmin(smiles)
            return int(a_star), int(a_star)
        except Exception as exc:  # pragma: no cover - error path
            # Catch both AssemblyMCError and ImportError from RDKit
            from .extern.assemblymc import AssemblyMCError

            if allow_fallback:
                return _surrogate_bounds(graph)
            if isinstance(exc, AssemblyMCError):
                raise
            raise AssemblyMCError(str(exc)) from exc

    raise ValueError(f"Unknown ai.method '{method}'")
