import random

from loto.scheduling.schedule_model import SchedGraph, SchedTask
from loto.scheduling.rework import maybe_rework


def _base_graph() -> SchedGraph:
    tasks = {
        "A": SchedTask(id="A"),
        "B": SchedTask(id="B", predecessors=("A",)),
    }
    return SchedGraph(tasks=tasks, resources={})


def test_rework_occurs_when_p_one() -> None:
    graph = _base_graph()
    alt = SchedTask(id="B", predecessors=("B",))
    transform = maybe_rework(1.0, alt, rng=random.Random(0))
    new_graph = transform(graph)

    assert new_graph is not graph
    assert set(new_graph.tasks) == {"A", "B", "B_1"}
    assert set(graph.tasks) == {"A", "B"}  # original unchanged

    new_graph.validate()


def test_rework_skipped_when_p_zero() -> None:
    graph = _base_graph()
    alt = SchedTask(id="B", predecessors=("B",))
    transform = maybe_rework(0.0, alt, rng=random.Random(0))
    same_graph = transform(graph)

    assert same_graph is graph
    assert set(graph.tasks) == {"A", "B"}

    same_graph.validate()
