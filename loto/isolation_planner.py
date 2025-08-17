from __future__ import annotations

"""Simple isolation planner for block and bleed selection.

This module implements a very small subset of the functionality required to
select isolation points for a given asset within a process graph.  It is not a
full featured planner; it only supports the pieces needed by the accompanying
tests.  The planner searches the graph for the nearest block valves, bleed
valves, drains and verification instrumentation (PT/TT) using hop count as the
primary metric and ``health_score`` as a tie breaker.
"""

from dataclasses import dataclass
from typing import List, Sequence

import networkx as nx

__all__ = ["IsolationPlan", "plan_isolation"]


@dataclass
class IsolationPlan:
    """Container for the isolation points returned by :func:`plan_isolation`."""

    blocks: List[str]
    bleed: str
    drain: str
    verify: str


def _choose(
    g: nx.MultiDiGraph,
    distances: dict[str, int],
    types: Sequence[str],
    count: int,
) -> List[str]:
    """Return ``count`` nodes of ``types`` sorted by distance then health.

    Parameters
    ----------
    g:
        Graph to inspect.
    distances:
        Mapping of node id to hop count from the asset.
    types:
        Acceptable node ``type`` values.
    count:
        Number of nodes to return.
    """

    candidates: List[tuple[int, float, str]] = []
    for node, attrs in g.nodes(data=True):
        if attrs.get("type") in types and node in distances:
            dist = distances[node]
            health = attrs.get("health_score", 0.0) or 0.0
            candidates.append((dist, -health, node))
    candidates.sort()
    return [n for _, _, n in candidates[:count]]


def plan_isolation(g: nx.MultiDiGraph, asset: str) -> IsolationPlan:
    """Plan isolation points for ``asset`` within ``g``.

    The algorithm is extremely small in scope:

    - two closest block valves
    - one closest bleed valve
    - nearest drain
    - nearest PT/TT for verification

    Nodes are compared first on hop count from ``asset`` and then by highest
    ``health_score``.  The graph is treated as undirected for distance
    calculations.
    """

    if asset not in g:
        raise ValueError("unknown asset node")

    undirected = g.to_undirected()
    distances = nx.single_source_shortest_path_length(undirected, asset)

    blocks = _choose(g, distances, ["block", "valve"], 2)
    if len(blocks) != 2:
        raise ValueError("expected at least two block valves")

    bleeds = _choose(g, distances, ["bleed", "vent"], 1)
    if not bleeds:
        raise ValueError("expected a bleed valve")

    drains = _choose(g, distances, ["drain"], 1)
    if not drains:
        raise ValueError("expected a drain")

    verifies = _choose(g, distances, ["PT", "TT"], 1)
    if not verifies:
        raise ValueError("expected a PT or TT for verification")

    return IsolationPlan(blocks=blocks, bleed=bleeds[0], drain=drains[0], verify=verifies[0])
