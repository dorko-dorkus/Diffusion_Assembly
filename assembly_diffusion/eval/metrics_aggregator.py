from __future__ import annotations

"""Utilities to aggregate evaluation metrics into CSV summaries.

baseline: simple molecule sets provide a sanity check for metric aggregation.
method: compute RDKit-based metrics for each named sample set and record counts
    such as valid and unique molecules.  Results are written as CSV rows.
objective: enable batch evaluation of multiple runs or baselines with
    reproducible metrics.
params: mapping of set names to ``MoleculeGraph`` sequences, reference SMILES,
    output CSV path and random seed.
repro: deterministic operations mirror those in :mod:`assembly_diffusion.eval.metrics`.
validation: ``tests/test_metrics_aggregator.py`` exercises the aggregator on a
    toy dataset, ensuring validity is in [0,1], counts sum correctly and
    uniqueness/novelty ratios fall within expected bounds.
"""

from pathlib import Path
import csv
from typing import Iterable, Mapping, Sequence, Dict, Any, List

from .metrics import Metrics, smiles_set
from .validity import sanitize_or_none
from ..graph import MoleculeGraph


def aggregate_metrics(
    sample_sets: Mapping[str, Sequence[MoleculeGraph]],
    reference_smiles: Iterable[str],
    out_csv: str | Path,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Evaluate ``sample_sets`` and write a CSV summary.

    Parameters
    ----------
    sample_sets:
        Mapping from set name to sequence of ``MoleculeGraph`` objects.
    reference_smiles:
        SMILES strings used as novelty reference.
    out_csv:
        Destination CSV file.  Parent directories are created if necessary.
    seed:
        Random seed forwarded to :func:`Metrics.evaluate`.

    Returns
    -------
    List[Dict[str, Any]]
        List of metric rows written to ``out_csv``.
    """

    rows: List[Dict[str, Any]] = []
    ref_set = set(reference_smiles)

    for name, graphs in sample_sets.items():
        metrics = Metrics.evaluate(graphs, ref_set, seed=seed)

        total = len(graphs)
        num_valid = sum(1 for g in graphs if sanitize_or_none(g) is not None)
        unique_smiles = smiles_set(graphs)
        num_unique = len(unique_smiles)
        num_novel = sum(1 for s in unique_smiles if s not in ref_set)

        row: Dict[str, Any] = {
            "name": name,
            "n_total": total,
            "n_valid": num_valid,
            "n_invalid": total - num_valid,
            "n_unique": num_unique,
            "n_novel": num_novel,
            **metrics,
        }
        rows.append(row)

    if rows:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows
