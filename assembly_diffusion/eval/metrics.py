"""Evaluation metrics for sets of molecular graphs."""

from __future__ import annotations

from typing import Iterable, Sequence, Set, List

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None

from ..graph import MoleculeGraph
from .validity import sanitize_or_none
from ..ai_surrogate import AISurrogate



def smiles_set(graphs: Iterable[MoleculeGraph]) -> Set[str]:
    """Return a set of canonical SMILES for valid graphs."""

    if Chem is None:
        raise ImportError("RDKit is required to generate SMILES")
    result: Set[str] = set()
    for g in graphs:
        mol = sanitize_or_none(g)
        if mol is not None:
            result.add(Chem.MolToSmiles(mol, canonical=True))
    return result


class Metrics:
    """Basic metrics for evaluating generated molecules."""

    @staticmethod
    def evaluate(
        sample_set: Sequence[MoleculeGraph],
        train_smiles: Iterable[str],
    ) -> dict[str, float]:
        """Return validity, uniqueness and novelty for ``sample_set``."""

        if Chem is None:
            raise ImportError("RDKit is required to evaluate metrics")
        total = len(sample_set)
        if total == 0:
            return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0}

        # Determine number of valid molecules
        num_valid = sum(1 for g in sample_set if sanitize_or_none(g) is not None)

        validity = num_valid / total
        unique_smiles = smiles_set(sample_set)
        uniqueness = len(unique_smiles) / num_valid if num_valid else 0.0
        train_set = set(train_smiles)
        novel = [s for s in unique_smiles if s not in train_set]
        novelty = len(novel) / len(unique_smiles) if unique_smiles else 0.0
        return {"validity": validity, "uniqueness": uniqueness, "novelty": novelty}


class AIMetrics:
    """Metrics for surrogate score trajectories."""

    @staticmethod
    def trajectory_scores(
        trajectories: Sequence[Sequence[MoleculeGraph]],
        surrogate: AISurrogate,
    ) -> List[List[float]]:
        """Return surrogate scores for each trajectory."""

        return [[surrogate.score(g) for g in traj] for traj in trajectories]

    @staticmethod
    def monotonicity(score_sequences: Sequence[Sequence[float]]) -> float:
        """Fraction of sequences with non-decreasing scores."""

        seqs = [s for s in score_sequences if len(s) > 1]
        if not seqs:
            return 0.0
        count = 0
        for seq in seqs:
            if all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)):
                count += 1
        return count / len(seqs)
