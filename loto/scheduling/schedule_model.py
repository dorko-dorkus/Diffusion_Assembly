from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
import heapq


@dataclass(frozen=True)
class SchedTask:
    """A single scheduled task with optional dependencies and resources."""

    id: str
    resources: Tuple[str, ...] = ()
    predecessors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Resource:
    """A consumable resource with positive capacity."""

    id: str
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("Resource capacity must be > 0")


@dataclass(frozen=True)
class Calendar:
    """Working time definition with timezone information."""

    tz: str
    work_intervals: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class SchedGraph:
    """Collection of tasks and resources with validation utilities."""

    tasks: Dict[str, SchedTask]
    resources: Dict[str, Resource]
    calendar: Calendar | None = None

    def validate(self) -> None:
        """Validate structural consistency of the scheduling graph."""

        # Check predecessor and resource references
        for task in self.tasks.values():
            for pred in task.predecessors:
                if pred not in self.tasks:
                    raise ValueError(f"Unknown predecessor '{pred}' for task '{task.id}'")
            for rid in task.resources:
                if rid not in self.resources:
                    raise ValueError(f"Unknown resource '{rid}' for task '{task.id}'")

        # Detect cycles via topological sort
        try:
            self.toposort()
        except ValueError as exc:  # pragma: no cover - toposort handles message
            raise ValueError("Cycle detected") from exc

    def toposort(self) -> List[SchedTask]:
        """Return tasks in a stable topological order."""

        indeg = {tid: 0 for tid in self.tasks}
        succ: Dict[str, List[str]] = defaultdict(list)

        for task in self.tasks.values():
            for pred in task.predecessors:
                indeg[task.id] += 1
                succ[pred].append(task.id)

        for lst in succ.values():
            lst.sort()

        heap: List[str] = [tid for tid, d in indeg.items() if d == 0]
        heapq.heapify(heap)

        ordered: List[SchedTask] = []
        indeg_local = indeg.copy()

        while heap:
            tid = heapq.heappop(heap)
            ordered.append(self.tasks[tid])
            for nxt in succ.get(tid, []):
                indeg_local[nxt] -= 1
                if indeg_local[nxt] == 0:
                    heapq.heappush(heap, nxt)

        if len(ordered) != len(self.tasks):
            raise ValueError("Cycle detected")

        return ordered


__all__ = [
    "SchedTask",
    "Resource",
    "Calendar",
    "SchedGraph",
]
