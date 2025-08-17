import pytest

from loto.graph_builder import Edge, Graph, Node, validate

def test_validate_clean():
    nodes = {
        "A": Node("A", "valve", "steam", orientation="in", valve_type="ball"),
        "B": Node("B", "pump", "steam"),
    }
    edges = [Edge("A", "B", direction="fwd")]
    graph = Graph(nodes, edges)
    assert validate([graph]) == []


def test_orphan_detection():
    nodes = {
        "A": Node("A", "valve", "steam", orientation="in", valve_type="ball"),
        "B": Node("B", "pump", "steam"),
        "C": Node("C", "sensor", "steam"),
    }
    edges = [Edge("A", "B", direction="fwd")]
    graph = Graph(nodes, edges)
    issues = validate([graph])
    assert [i.code for i in issues] == ["GRAPH/ORPHAN"]


def test_missing_direction():
    nodes = {
        "A": Node("A", "valve", "steam", orientation="in", valve_type="ball"),
        "B": Node("B", "pump", "steam"),
    }
    edges = [Edge("A", "B")]
    graph = Graph(nodes, edges)
    issues = validate([graph])
    assert [i.code for i in issues] == ["GRAPH/NO_DIRECTION"]


def test_cross_domain_without_exchanger():
    nodes = {
        "A": Node("A", "valve", "steam", orientation="in", valve_type="ball"),
        "B": Node("B", "pump", "condensate"),
    }
    edges = [Edge("A", "B", direction="fwd")]
    graph = Graph(nodes, edges)
    issues = validate([graph])
    assert [i.code for i in issues] == ["GRAPH/CROSS_DOMAIN"]


def test_dangling_iso_tag():
    nodes = {
        "A": Node("A", "valve", "steam", iso_tag="ISO1", orientation="in", valve_type="ball"),
        "B": Node("B", "pump", "steam"),
    }
    edges = [Edge("A", "B", direction="fwd")]
    graph = Graph(nodes, edges)
    issues = validate([graph])
    assert [i.code for i in issues] == ["GRAPH/DANGLING_ISO"]


def test_unknown_valve_orientation():
    nodes = {
        "A": Node("A", "valve", "steam", orientation="sideways", valve_type="ball"),
        "B": Node("B", "pump", "steam"),
    }
    edges = [Edge("A", "B", direction="fwd")]
    graph = Graph(nodes, edges)
    issues = validate([graph])
    assert [i.code for i in issues] == ["GRAPH/UNKNOWN_VALVE"]
