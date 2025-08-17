"""Utilities to construct NetworkX graphs from CSV inputs."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping

import networkx as nx

__all__ = ["from_csvs"]


def _trim_mapping(row: Mapping[str, str]) -> Dict[str, str]:
    """Return a new dict with whitespace trimmed from string values."""
    out: Dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[key] = value.strip()
        else:
            out[key] = value
    return out


_TRUTHY = {"true", "t", "1", "yes", "y"}
_FALSY = {"false", "f", "0", "no", "n", ""}


def _to_bool(value: str) -> bool:
    """Convert a CSV field to boolean in a case-insensitive manner."""
    val = (value or "").strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def from_csvs(nodes_csv: Path, edges_csv: Path) -> Dict[str, nx.MultiDiGraph]:
    """Build graphs grouped by domain from node and edge CSV files.

    Parameters
    ----------
    nodes_csv:
        Path to a CSV file describing nodes. Required columns: ``tag``,
        ``type``, ``domain``, ``fail_state`` and ``health_score``.
    edges_csv:
        Path to a CSV file describing edges. Required columns: ``from_tag``,
        ``to_tag``, ``domain`` and edge attributes ``is_isolation_point``,
        ``iso_tag``, ``direction``, ``size_mm`` and ``bypass_group``.

    Returns
    -------
    Dict[str, nx.MultiDiGraph]
        Mapping of domain name to constructed graph.
    """

    graphs: Dict[str, nx.MultiDiGraph] = defaultdict(nx.MultiDiGraph)

    # --- Load nodes ---
    node_rows = []
    with open(Path(nodes_csv), newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            row = _trim_mapping(raw_row)
            node_rows.append(row)

    tags = [r["tag"] for r in node_rows]
    if len(tags) != len(set(tags)):
        raise ValueError("Duplicate node tags encountered")

    node_rows.sort(key=lambda r: (r.get("domain", ""), r["tag"]))

    for row in node_rows:
        domain = row.get("domain", "")
        g = graphs[domain]
        attrs = {
            "type": row.get("type"),
            "domain": domain,
            "tag": row.get("tag"),
            "fail_state": row.get("fail_state"),
            "health_score": float(row["health_score"]) if row.get("health_score") else None,
        }
        g.add_node(row["tag"], **attrs)

    # --- Load edges ---
    edge_rows = []
    with open(Path(edges_csv), newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            row = _trim_mapping(raw_row)
            edge_rows.append(row)

    edge_rows.sort(
        key=lambda r: (r.get("domain", ""), r.get("from_tag", ""), r.get("to_tag", ""), r.get("iso_tag", ""))
    )

    for row in edge_rows:
        domain = row.get("domain", "")
        g = graphs[domain]
        is_iso = _to_bool(row.get("is_isolation_point", "false"))
        attrs = {
            "is_isolation_point": is_iso,
            "iso_tag": row.get("iso_tag"),
            "direction": row.get("direction"),
            "size_mm": float(row["size_mm"]) if row.get("size_mm") else None,
            "bypass_group": row.get("bypass_group"),
        }
        g.add_edge(row.get("from_tag"), row.get("to_tag"), **attrs)

    return dict(graphs)
