from __future__ import annotations

import networkx as nx

from loto.isolation_planner import IsolationPlan
from loto import sim_engine


def _build_graphs():
    g = nx.MultiDiGraph()
    # asset and other nodes
    g.add_node("A", type="asset")
    g.add_node("B1", type="block", iso_tag="B1")
    g.add_node("B2", type="block", iso_tag="B2")
    g.add_node("L1", type="bleed", iso_tag="L1")
    g.add_node("D1", type="drain", iso_tag="D1")
    g.add_node("C", type="pipe")

    # block valve edges
    g.add_edge("A", "B1", iso_tag="B1")
    g.add_edge("B1", "A", iso_tag="B1")
    g.add_edge("B1", "B2", iso_tag="B2")
    g.add_edge("B2", "B1", iso_tag="B2")

    # normal connection
    g.add_edge("B2", "C")
    g.add_edge("C", "B2")

    # drain and bleed connections
    g.add_edge("A", "L1", iso_tag="L1")
    g.add_edge("A", "D1", iso_tag="D1")

    other = nx.MultiDiGraph()
    other.add_edge("X", "Y")

    return {"steam": g, "cond": other}


def test_apply_plan_removes_blocks_and_opens_drains():
    graphs = _build_graphs()
    plan = IsolationPlan(blocks=["B1", "B2"], bleed="L1", drain="D1", verify="PT1")

    result = sim_engine.apply_plan(graphs, plan)

    # original graphs untouched
    assert graphs["steam"].number_of_edges() == 8
    assert "state" not in graphs["steam"].nodes["L1"]
    orig_iso = {d.get("iso_tag") for _, _, d in graphs["steam"].edges(data=True)}
    assert "B1" in orig_iso and "B2" in orig_iso

    # block edges removed in result
    steam = result["steam"]
    iso_tags = {data.get("iso_tag") for _, _, data in steam.edges(data=True)}
    assert "B1" not in iso_tags and "B2" not in iso_tags

    # drain and bleed marked open
    assert steam.nodes["L1"]["state"] == "open"
    assert steam.nodes["D1"]["state"] == "open"

    # other domain preserved
    assert list(result["cond"].edges()) == list(graphs["cond"].edges())
