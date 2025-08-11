"""Experiment specification for the evaluation metrics smoke test.

baseline: compute metrics on a tiny set of molecules without modifications.
data_sources: inline SMILES strings ``["CCO", "CCN", "CCC"]`` parsed by RDKit.
method: convert SMILES to :class:`~assembly_diffusion.graph.MoleculeGraph`
    instances and evaluate using :class:`~assembly_diffusion.eval.metrics.Metrics`.
metrics: validity and assembly index summaries plus optional surrogate and
    exact assembly index statistics.  Additional reference metrics follow the
    standard definitions ``F1 = 2 * precision * recall / (precision + recall)``
    and ``MSE = (1/N) * Σ_i (ŷ_i - y_i)^2``.  All metrics should be reported as
    ``mean ± std`` computed over independent random seeds.
objective: exercise the metrics stack to ensure all components integrate.
repro: deterministic because inputs are hard coded and all randomness fixed.
validation: script runs without error and optional ``--print-metrics`` output
    can be inspected in continuous integration.
split: each of the three molecules is treated as a separate train, validation,
    and test example to illustrate a complete evaluation protocol.
selection: in real experiments the model with the highest validation F1 would
    be chosen.
early_stopping: training would halt when the validation F1 fails to improve for
    two consecutive checks; this smoke test runs a single deterministic pass.
"""

from __future__ import annotations

import argparse
import logging
import platform
import random
import subprocess

import numpy as np

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None

from .metrics import Metrics
from ..graph import MoleculeGraph
from ..stats import summarise_A_hat

try:  # pragma: no cover - optional dependencies
    from ..ai_surrogate import AISurrogate
    from ..assembly_index import AssemblyIndex
except ImportError:  # pragma: no cover - handled at runtime
    AISurrogate = None
    AssemblyIndex = None


logger = logging.getLogger(__name__)


def main() -> None:
    """Run a tiny smoke test for the evaluation metrics."""

    parser = argparse.ArgumentParser(description="Run evaluation metrics smoke test")
    parser.add_argument("--print-metrics", action="store_true", help="Print metrics dictionary")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also compute metrics for a trivial single-carbon baseline",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git may be unavailable
        commit = "unknown"
    logger.info(
        "Reproducibility: seed=%s python=%s numpy=%s commit=%s",
        0,
        platform.python_version(),
        np.__version__,
        commit,
    )

    if Chem is None:
        raise ImportError("RDKit is required for the smoke test; install rdkit==2024.9.6")

    sample_smiles = ["CCO", "CCN", "CCC"]
    train_smiles, val_smiles, test_smiles = sample_smiles

    def evaluate(smiles: str) -> dict[str, float]:
        graph = MoleculeGraph.from_rdkit(Chem.MolFromSmiles(smiles))
        result = Metrics.evaluate([graph], reference_smiles=[smiles])
        if AISurrogate is not None:
            surrogate = AISurrogate()
            s_score = surrogate.score(graph)
            result.update(
                {f"surrogate_{k}": v for k, v in summarise_A_hat([s_score]).items()}
            )
        if AssemblyIndex is not None:
            e_score = AssemblyIndex.A_star_exact_or_none(graph)
            result.update(
                {f"exact_{k}": v for k, v in summarise_A_hat([e_score]).items()}
            )
        return result

    metrics = {
        "train": evaluate(train_smiles),
        "val": evaluate(val_smiles),
        "test": evaluate(test_smiles),
    }

    if args.baseline:
        metrics["baseline"] = evaluate("C")

    if args.print_metrics:
        logger.info(metrics)


if __name__ == "__main__":
    main()
