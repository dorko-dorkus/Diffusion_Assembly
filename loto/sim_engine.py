from __future__ import annotations

from typing import Dict

import networkx as nx

from .isolation_planner import IsolationPlan

__all__ = ["apply_plan"]


def apply_plan(
    graphs: Dict[str, nx.MultiDiGraph],
    plan: IsolationPlan,
) -> Dict[str, nx.MultiDiGraph]:
    """Apply ``plan`` to ``graphs`` and return new per-domain graphs.

    The function performs a purely functional transformation – ``graphs`` are
    left unmodified.  Selected block valves are isolated by removing all edges
    whose ``iso_tag`` matches the ``iso_tag`` of a block valve from the plan.
    Drain and bleed nodes referenced by the plan are marked with
    ``state='open'``.  All remaining topology is preserved.
    """

    # Clone graphs to avoid mutating the originals
    cloned: Dict[str, nx.MultiDiGraph] = {
        domain: g.copy() for domain, g in graphs.items()
    }

    # Determine iso_tag values for the block valves
    block_iso_tags: set[str] = set()
    for node in plan.blocks:
        for g in cloned.values():
            if node in g and g.nodes[node].get("iso_tag"):
                block_iso_tags.add(g.nodes[node]["iso_tag"])

    # Remove edges corresponding to selected blocks
    for g in cloned.values():
        to_remove = []
        for u, v, key, data in g.edges(keys=True, data=True):
            if data.get("iso_tag") in block_iso_tags:
                to_remove.append((u, v, key))
        if to_remove:
            g.remove_edges_from(to_remove)

        # Mark drain and bleed nodes open where present
        for node in (plan.bleed, plan.drain):
            if node in g:
                g.nodes[node]["state"] = "open"

    return cloned
