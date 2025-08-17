from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "GraphError",
    "Issue",
    "Node",
    "Edge",
    "Graph",
    "validate",
]


class GraphError(Exception):
    """Exception raised for fatal graph construction errors."""

    pass


@dataclass(frozen=True)
class Issue:
    """Validation issue reported for a graph."""

    code: str
    message: str
    ref: str


@dataclass
class Node:
    """Node within a simple directed graph."""

    id: str
    type: str
    domain: Optional[str] = None
    iso_tag: Optional[str] = None
    orientation: Optional[str] = None
    valve_type: Optional[str] = None


@dataclass
class Edge:
    """Directed edge connecting two nodes."""

    src: str
    dst: str
    direction: Optional[str] = None


@dataclass
class Graph:
    """Container for nodes and edges."""

    nodes: Dict[str, Node]
    edges: List[Edge]


def validate(graphs: List[Graph]) -> List[Issue]:
    """Validate graphs and return a list of discovered issues.

    The function performs lightweight structural checks:
    - orphan nodes (no incident edges)
    - missing edge direction
    - cross-domain hops without an exchanger node
    - dangling ``iso_tag`` values (referenced exactly once)
    - unknown valve orientation or type

    Parameters
    ----------
    graphs:
        A list of :class:`Graph` objects to validate.

    Returns
    -------
    list[Issue]
        All issues discovered across the supplied graphs.
    """

    issues: List[Issue] = []
    allowed_orientations = {"in", "out"}
    allowed_valve_types = {"ball", "gate"}

    for gi, g in enumerate(graphs):
        connected: set[str] = set()
        iso_counts: Dict[str, int] = {}

        for node in g.nodes.values():
            if node.iso_tag:
                iso_counts[node.iso_tag] = iso_counts.get(node.iso_tag, 0) + 1

        for edge in g.edges:
            if not edge.direction:
                issues.append(
                    Issue(
                        code="GRAPH/NO_DIRECTION",
                        message=f"edge {edge.src}-{edge.dst} missing direction",
                        ref=f"{gi}:{edge.src}-{edge.dst}",
                    )
                )
            connected.update([edge.src, edge.dst])
            src = g.nodes.get(edge.src)
            dst = g.nodes.get(edge.dst)
            if src and dst:
                if src.domain and dst.domain and src.domain != dst.domain:
                    if src.type != "exchanger" and dst.type != "exchanger":
                        issues.append(
                            Issue(
                                code="GRAPH/CROSS_DOMAIN",
                                message=(
                                    f"{edge.src}->{edge.dst} crosses {src.domain}->{dst.domain}"
                                ),
                                ref=f"{gi}:{edge.src}->{edge.dst}",
                            )
                        )

        # Orphan nodes
        for nid in set(g.nodes) - connected:
            issues.append(
                Issue(
                    code="GRAPH/ORPHAN",
                    message=f"node {nid} has no connections",
                    ref=f"{gi}:{nid}",
                )
            )

        # Dangling iso tags
        for iso, count in iso_counts.items():
            if count == 1:
                node_id = next(n.id for n in g.nodes.values() if n.iso_tag == iso)
                issues.append(
                    Issue(
                        code="GRAPH/DANGLING_ISO",
                        message=f"iso_tag {iso} on {node_id} is unused",
                        ref=f"{gi}:{node_id}",
                    )
                )

        # Valve validation
        for node in g.nodes.values():
            if node.type == "valve":
                if node.orientation and node.orientation not in allowed_orientations:
                    issues.append(
                        Issue(
                            code="GRAPH/UNKNOWN_VALVE",
                            message=f"unknown orientation '{node.orientation}'",
                            ref=f"{gi}:{node.id}",
                        )
                    )
                if node.valve_type and node.valve_type not in allowed_valve_types:
                    issues.append(
                        Issue(
                            code="GRAPH/UNKNOWN_VALVE",
                            message=f"unknown type '{node.valve_type}'",
                            ref=f"{gi}:{node.id}",
                        )
                    )

    return issues
