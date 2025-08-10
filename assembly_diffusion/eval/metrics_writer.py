"""Experiment specification for serialising experiment metrics.

data_sources: metric dictionaries produced by experiment runs and model
    evaluations.
method: normalise numeric values, add a ``schema_version`` field and dump a
    ``metrics.json`` file in the output directory.
objective: persist metrics in a consistent JSON format for downstream analysis
    and comparison.
params: ``outdir`` destination directory and ``**metrics`` arbitrary name/value
    pairs where numbers are cast to ``float`` or ``int``.
repro: serialisation is deterministic; identical inputs yield identical
    ``metrics.json`` files.
validation: tests or downstream consumers can load ``metrics.json`` and check
    for expected keys and numeric types.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.1"


def write_metrics(outdir: str, **metrics: Any) -> None:
    """Write ``metrics.json`` for an experiment run.

    Historically this helper accepted a fixed set of keyword-only arguments for
    the handful of metrics that were available.  As the project grew it became
    useful to store additional statistics, for example confidence interval (CI)
    bounds for ablation studies.  To keep the call site flexible the function
    now accepts arbitrary keyword arguments and simply serialises them to
    ``metrics.json``.

    Parameters
    ----------
    outdir:
        Directory in which ``metrics.json`` will be written.  The directory is
        created if necessary.
    **metrics:
        Mapping of metric names to values.  Numeric values are cast to ``float``
        for JSON serialisation; other types (e.g. lists for CIs) are stored as
        provided.
    """

    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key, value in metrics.items():
        # Preserve ints (e.g., random seeds) while normalising floats.
        if isinstance(value, float):
            payload[key] = float(value)
        elif isinstance(value, int):
            payload[key] = int(value)
        else:
            payload[key] = value

    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
