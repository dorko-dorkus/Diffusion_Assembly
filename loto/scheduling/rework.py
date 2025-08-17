from __future__ import annotations

import random
from typing import Callable

from .schedule_model import SchedGraph, SchedTask


def maybe_rework(p: float, alt_task: SchedTask, rng: random.Random | None = None) -> Callable[[SchedGraph], SchedGraph]:
    """Conditionally inject a rework task into a scheduling graph.

    Parameters
    ----------
    p:
        Probability of inserting ``alt_task``.  Must be in [0, 1].
    alt_task:
        Task template for the rework branch.  Its ``id`` will be made unique
        before insertion, but its ``predecessors`` are preserved.
    rng:
        Optional random number generator.  Defaults to :mod:`random`'s module
        level generator.

    Returns
    -------
    Callable[[SchedGraph], SchedGraph]
        Function that takes a :class:`~loto.scheduling.schedule_model.SchedGraph`
        and returns a possibly augmented copy.
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be within [0, 1]")

    rng = rng or random

    def _inject(graph: SchedGraph) -> SchedGraph:
        if rng.random() >= p:
            return graph

        # Copy tasks to avoid mutating the input graph
        tasks = dict(graph.tasks)

        base_id = alt_task.id
        new_id = base_id
        suffix = 1
        while new_id in tasks:
            new_id = f"{base_id}_{suffix}"
            suffix += 1

        tasks[new_id] = SchedTask(id=new_id,
                                  resources=alt_task.resources,
                                  predecessors=alt_task.predecessors)

        new_graph = SchedGraph(tasks=tasks,
                               resources=dict(graph.resources),
                               calendar=graph.calendar)
        # Ensure the resulting graph is structurally sound
        new_graph.validate()
        return new_graph

    return _inject


__all__ = ["maybe_rework"]
