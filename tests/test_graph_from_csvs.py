import csv
from pathlib import Path

import pytest

from loto.graph_builder import from_csvs


@pytest.fixture
def sample_csvs(tmp_path: Path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"

    # deliberately unsorted and with whitespace
    node_rows = [
        {"tag": "B ", "type": "Asset", "domain": "steam", "fail_state": "closed", "health_score": "0.8"},
        {"tag": " A", "type": "Asset", "domain": "steam", "fail_state": "open", "health_score": "0.9"},
        {"tag": "C", "type": "Asset", "domain": "air", "fail_state": "open", "health_score": "0.95"},
        {"tag": "D", "type": "Asset", "domain": "air", "fail_state": "open", "health_score": "0.7"},
    ]
    edge_rows = [
        {
            "from_tag": "C",
            "to_tag": " D",
            "domain": "air",
            "is_isolation_point": "false",
            "iso_tag": "ISO2 ",
            "direction": "return",
            "size_mm": "50",
            "bypass_group": "",
        },
        {
            "from_tag": " A",
            "to_tag": " B ",
            "domain": "steam",
            "is_isolation_point": "TRUE",
            "iso_tag": " ISO1",
            "direction": "supply",
            "size_mm": "100",
            "bypass_group": "BG1",
        },
    ]

    with nodes_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["tag", "type", "domain", "fail_state", "health_score"]
        )
        writer.writeheader()
        writer.writerows(node_rows)

    with edges_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "from_tag",
                "to_tag",
                "domain",
                "is_isolation_point",
                "iso_tag",
                "direction",
                "size_mm",
                "bypass_group",
            ],
        )
        writer.writeheader()
        writer.writerows(edge_rows)

    return nodes_path, edges_path


def test_build_graphs(sample_csvs):
    nodes_csv, edges_csv = sample_csvs
    graphs = from_csvs(nodes_csv, edges_csv)

    assert set(graphs.keys()) == {"steam", "air"}

    steam = graphs["steam"]
    air = graphs["air"]

    assert list(steam.nodes) == ["A", "B"]  # deterministic order
    assert list(air.nodes) == ["C", "D"]

    attrs_a = steam.nodes["A"]
    assert attrs_a["type"] == "Asset"
    assert attrs_a["domain"] == "steam"
    assert attrs_a["fail_state"] == "open"
    assert attrs_a["health_score"] == pytest.approx(0.9)

    edge_data = list(steam.edges(data=True))
    assert edge_data == [
        (
            "A",
            "B",
            {
                "is_isolation_point": True,
                "iso_tag": "ISO1",
                "direction": "supply",
                "size_mm": 100.0,
                "bypass_group": "BG1",
            },
        )
    ]

    air_edge = list(air.edges(data=True))[0][2]
    assert air_edge["is_isolation_point"] is False
    assert air_edge["iso_tag"] == "ISO2"


def test_duplicate_tags(tmp_path: Path):
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"

    with nodes_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["tag", "type", "domain", "fail_state", "health_score"]
        )
        writer.writeheader()
        writer.writerow(
            {"tag": "N1", "type": "Asset", "domain": "steam", "fail_state": "open", "health_score": "1"}
        )
        writer.writerow(
            {"tag": "N1", "type": "Asset", "domain": "steam", "fail_state": "open", "health_score": "1"}
        )

    with edges_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "from_tag",
                "to_tag",
                "domain",
                "is_isolation_point",
                "iso_tag",
                "direction",
                "size_mm",
                "bypass_group",
            ],
        )
        writer.writeheader()

    with pytest.raises(ValueError):
        from_csvs(nodes_csv, edges_csv)
