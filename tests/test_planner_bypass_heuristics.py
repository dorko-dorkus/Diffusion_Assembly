import networkx as nx

from loto import isolation_planner


def test_bypass_and_lockable_preference():
    g = nx.MultiDiGraph()
    g.add_node("A", type="asset", health_score=1.0)
    g.add_node("B1", type="block", health_score=0.5, lockable=True)
    g.add_node("B2", type="block", health_score=0.6, lockable=True)
    g.add_node("B3", type="block", health_score=0.7, lockable=False)
    g.add_node("L1", type="bleed", health_score=0.9, lockable=False)
    g.add_node("L2", type="bleed", health_score=0.8, lockable=True)
    g.add_node("D1", type="drain", health_score=0.4)
    g.add_node("PT1", type="PT", health_score=0.95)

    g.add_edge("A", "B1")
    g.add_edge("B1", "B2", bypass_group="BG1")
    g.add_edge("B1", "B3", bypass_group="BG1")
    g.add_edge("B3", "B2", bypass_group="BG1")

    g.add_edge("A", "L1")
    g.add_edge("A", "L2")
    g.add_edge("A", "D1")
    g.add_edge("A", "PT1")

    plan = isolation_planner.plan_isolation(g, "A")

    assert plan.blocks == ["B1", "B2", "B3"]
    assert plan.bleed == "L2"
    assert plan.drain == "D1"
    assert plan.verify == "PT1"
