import networkx as nx

from loto import isolation_planner


def _distance(g: nx.MultiDiGraph, src: str, dst: str) -> int:
    return nx.shortest_path_length(g.to_undirected(), src, dst)


def test_steam_ddbb_with_drain_and_verify():
    g = nx.MultiDiGraph()
    # asset node
    g.add_node("A", type="asset", health_score=1.0)
    # block valves
    g.add_node("B1", type="block", health_score=0.5)
    g.add_node("B2", type="block", health_score=0.9)
    g.add_node("B3", type="block", health_score=0.7)
    # bleeds
    g.add_node("L1", type="bleed", health_score=0.6)
    g.add_node("L2", type="bleed", health_score=0.8)
    # drain and verification instrumentation
    g.add_node("D1", type="drain", health_score=0.4)
    g.add_node("PT1", type="PT", health_score=0.95)

    # edges connecting the network
    g.add_edge("A", "B1")
    g.add_edge("B1", "B2")
    g.add_edge("B1", "B3")
    g.add_edge("A", "L1")
    g.add_edge("A", "L2")
    g.add_edge("A", "D1")
    g.add_edge("A", "PT1")

    plan = isolation_planner.plan_isolation(g, "A")

    # Expect two blocks and a bleed
    assert plan.blocks == ["B1", "B2"]
    assert plan.bleed == "L2"  # L2 preferred over L1 due to higher health

    # Drain and verification points present
    assert plan.drain == "D1"
    assert plan.verify == "PT1"

    # Ensure the selected points are close to the asset
    assert _distance(g, "A", plan.blocks[0]) == 1
    assert _distance(g, "A", plan.blocks[1]) == 2
    assert _distance(g, "A", plan.bleed) == 1
    assert _distance(g, "A", plan.drain) == 1
    assert _distance(g, "A", plan.verify) == 1
