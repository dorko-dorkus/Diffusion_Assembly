from __future__ import annotations

import argparse

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None

from .metrics import Metrics
from ..graph import MoleculeGraph


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
    if args.print_metrics:
        print(metrics)


if __name__ == "__main__":
    main()
