from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, List, Optional

import networkx as nx

__all__ = [
    "GraphError",
    "Issue",
    "Node",
    "Edge",
    "Graph",
    "from_csvs",
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


def from_csvs(node_csv: Path | str, edge_csv: Path | str) -> Dict[str, nx.MultiDiGraph]:
    """Construct graphs for each domain from node and edge CSV files.

    Parameters
    ----------
    node_csv:
        Path to a CSV containing node definitions. Required columns are
        ``tag``, ``type``, ``domain``, ``fail_state`` and ``health_score``.
    edge_csv:
        Path to a CSV containing edge definitions. Required columns are
        ``from_tag``, ``to_tag``, ``is_isolation_point``, ``iso_tag``,
        ``direction``, ``size_mm`` and ``bypass_group``.

    Returns
    -------
    dict[str, nx.MultiDiGraph]
        A mapping of domain name to graph constructed for that domain.

    Notes
    -----
    - Leading and trailing whitespace is trimmed from all fields.
    - Boolean flags are interpreted case-insensitively.
    - Duplicate node ``tag`` values raise :class:`GraphError`.
    """

    def _strip(val: Optional[str]) -> str:
        return val.strip() if val is not None else ""

    def _to_bool(val: str) -> bool:
        return _strip(val).lower() in {"1", "true", "yes", "y"}

    node_path = Path(node_csv)
    edge_path = Path(edge_csv)

    graphs: Dict[str, nx.MultiDiGraph] = {}
    nodes_by_tag: Dict[str, Dict[str, object]] = {}

    with node_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: _strip(r.get("tag", "")))
    for row in rows:
        tag = _strip(row.get("tag"))
        if not tag:
            continue
        if tag in nodes_by_tag:
            raise GraphError(f"duplicate node tag: {tag}")
        domain = _strip(row.get("domain")) or None
        node_attrs = {
            "type": _strip(row.get("type")) or None,
            "domain": domain,
            "tag": tag,
            "fail_state": _strip(row.get("fail_state")) or None,
            "health_score": float(_strip(row.get("health_score")))
            if _strip(row.get("health_score"))
            else None,
        }
        nodes_by_tag[tag] = node_attrs
        g = graphs.setdefault(domain or "", nx.MultiDiGraph())
        g.add_node(tag, **node_attrs)

    with edge_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(
        key=lambda r: (
            _strip(r.get("from_tag", "")),
            _strip(r.get("to_tag", "")),
            _strip(r.get("iso_tag", "")),
        )
    )
    for row in rows:
        src = _strip(row.get("from_tag"))
        dst = _strip(row.get("to_tag"))
        if not src or not dst:
            continue
        if src not in nodes_by_tag or dst not in nodes_by_tag:
            raise GraphError(f"edge references unknown node {src}->{dst}")
        src_dom = nodes_by_tag[src]["domain"]
        dst_dom = nodes_by_tag[dst]["domain"]
        if src_dom != dst_dom:
            raise GraphError(f"edge crosses domains {src_dom}->{dst_dom}")
        g = graphs.setdefault(src_dom or "", nx.MultiDiGraph())
        edge_attrs = {
            "is_isolation_point": _to_bool(row.get("is_isolation_point", "")),
            "iso_tag": _strip(row.get("iso_tag")) or None,
            "direction": _strip(row.get("direction")) or None,
            "size_mm": float(_strip(row.get("size_mm")))
            if _strip(row.get("size_mm"))
            else None,
            "bypass_group": _strip(row.get("bypass_group")) or None,
        }
        g.add_edge(src, dst, **edge_attrs)

    return graphs


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
