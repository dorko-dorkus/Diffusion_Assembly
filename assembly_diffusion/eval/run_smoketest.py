from __future__ import annotations

import argparse
import logging

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
    args = parser.parse_args()

    if Chem is None:
        raise ImportError("RDKit is required for the smoke test")

    sample_smiles = ["CCO", "CCN", "CCC"]
    graphs = [MoleculeGraph.from_rdkit(Chem.MolFromSmiles(s)) for s in sample_smiles]

    metrics = Metrics.evaluate(graphs, reference_smiles=sample_smiles)

    if AISurrogate is not None:
        surrogate = AISurrogate()
        s_scores = [surrogate.score(g) for g in graphs]
        metrics.update(
            {f"surrogate_{k}": v for k, v in summarise_A_hat(s_scores).items()}
        )

    if AssemblyIndex is not None:
        e_scores = [AssemblyIndex.A_star_exact_or_none(g) for g in graphs]
        metrics.update({f"exact_{k}": v for k, v in summarise_A_hat(e_scores).items()})

    if args.print_metrics:
        logger.info(metrics)


if __name__ == "__main__":
    main()
