import pytest

from loto.scheduling.schedule_model import (
    Calendar,
    Resource,
    SchedGraph,
    SchedTask,
)


def _build_valid_graph() -> SchedGraph:
    resources = {
        "R1": Resource(id="R1", capacity=1),
        "R2": Resource(id="R2", capacity=2),
    }
    tasks = {
        "A": SchedTask(id="A", resources=("R1",), predecessors=()),
        "B": SchedTask(id="B", resources=("R1",), predecessors=("A",)),
        "C": SchedTask(id="C", resources=("R2",), predecessors=("A",)),
        "D": SchedTask(id="D", resources=("R1", "R2"), predecessors=("B", "C")),
    }
    cal = Calendar(tz="UTC", work_intervals=(("09:00", "17:00"),))
    return SchedGraph(tasks=tasks, resources=resources, calendar=cal)


def test_valid_graph_passes() -> None:
    graph = _build_valid_graph()
    graph.validate()


def test_toposort_respects_dependencies() -> None:
    graph = _build_valid_graph()
    order = [task.id for task in graph.toposort()]
    assert order == ["A", "B", "C", "D"]


def test_cycle_triggers_error() -> None:
    resources = {"R1": Resource(id="R1", capacity=1)}
    tasks = {
        "A": SchedTask(id="A", resources=("R1",), predecessors=("C",)),
        "B": SchedTask(id="B", resources=("R1",), predecessors=("A",)),
        "C": SchedTask(id="C", resources=("R1",), predecessors=("B",)),
    }
    graph = SchedGraph(tasks=tasks, resources=resources)
    with pytest.raises(ValueError, match="Cycle detected"):
        graph.validate()
