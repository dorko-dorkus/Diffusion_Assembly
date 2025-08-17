from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from loto.graph_builder import GraphError, from_csvs


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_from_csvs_builds_domain_graphs(tmp_path: Path) -> None:
    nodes_csv = """tag,type,domain,fail_state,health_score
    A , valve , steam , closed , 0.9
    B , pump , steam , , 0.8
    HX1 , exchanger , condensate , , 1.0
    """

    edges_csv = """from_tag,to_tag,is_isolation_point,iso_tag,direction,size_mm,bypass_group
    A , B , TrUe , ISO-1 , fwd , 100 , BG1
    """

    node_path = _write(tmp_path, "nodes.csv", nodes_csv)
    edge_path = _write(tmp_path, "edges.csv", edges_csv)

    graphs = from_csvs(node_path, edge_path)

    assert set(graphs) == {"steam", "condensate"}
    steam = graphs["steam"]
    assert isinstance(steam, nx.MultiDiGraph)
    assert steam.number_of_nodes() == 2
    assert steam.number_of_edges() == 1

    n_attrs = steam.nodes["A"]
    assert n_attrs["type"] == "valve"
    assert n_attrs["fail_state"] == "closed"
    assert n_attrs["health_score"] == 0.9

    e_attrs = steam.get_edge_data("A", "B")[0]
    assert e_attrs["is_isolation_point"] is True
    assert e_attrs["iso_tag"] == "ISO-1"
    assert e_attrs["direction"] == "fwd"
    assert e_attrs["size_mm"] == 100.0
    assert e_attrs["bypass_group"] == "BG1"


def test_from_csvs_duplicate_node_tag(tmp_path: Path) -> None:
    nodes_csv = """tag,type,domain,fail_state,health_score
    A,valve,steam,closed,0.9
    A,pump,steam,,0.8
    """

    edges_csv = """from_tag,to_tag,is_isolation_point,iso_tag,direction,size_mm,bypass_group
    A,A,false,,fwd,100,
    """

    node_path = _write(tmp_path, "nodes.csv", nodes_csv)
    edge_path = _write(tmp_path, "edges.csv", edges_csv)

    with pytest.raises(GraphError):
        from_csvs(node_path, edge_path)

