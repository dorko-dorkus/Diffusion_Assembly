"""Evaluation metrics for sets of molecular graphs."""

from __future__ import annotations

SCHEMA_VERSION = "1.1.0"

from typing import Iterable, Sequence, Set, List, Any, Tuple

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from ..qed_sa import qed_sa_distribution
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


def _ecfp4_fingerprints(valid_smiles: List[str]) -> List[Any]:
    """Return RDKit ECFP4 fingerprints for ``valid_smiles``."""

    fps: List[Any] = []
    for s in valid_smiles:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def _mean_pairwise_tanimoto_distance(fps: List[Any]) -> float:
    """Return the mean pairwise ``1 - Tanimoto`` distance for fingerprints."""

    if len(fps) < 2:
        return 0.0
    n = len(fps)
    acc = 0.0
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            acc += 1.0 - sim
            k += 1
    return acc / k if k else 0.0


class Metrics:
    """Basic metrics for evaluating generated molecules."""

    @staticmethod
    def evaluate(
        sample_set: Sequence[MoleculeGraph],
        reference_smiles: Iterable[str],
    ) -> dict[str, float]:
        """Return RDKit-based metrics for ``sample_set``.

        The returned dictionary contains ``validity``, ``uniqueness``,
        ``diversity`` (legacy alias for ``diversity_ecfp4_mean_distance``),
        ``diversity_ecfp4_mean_distance`` (mean pairwise 1 - Tanimoto distance
        over ECFP4 fingerprints), ``novelty`` against the provided
        ``reference_smiles``, and distribution summaries for RDKit ``qed``
        and synthetic accessibility ``sa`` scores.
        """

        if Chem is None:
            raise ImportError("RDKit is required to evaluate metrics")
        total = len(sample_set)
        if total == 0:
            return {
                "validity": 0.0,
                "uniqueness": 0.0,
                "diversity": 0.0,
                "diversity_ecfp4_mean_distance": 0.0,
                "novelty": 0.0,
                "qed_mean": 0.0,
                "qed_median": 0.0,
                "sa_mean": 0.0,
                "sa_median": 0.0,
            }

        # Determine number of valid molecules
        num_valid = sum(1 for g in sample_set if sanitize_or_none(g) is not None)

        validity = num_valid / total
        unique_smiles = smiles_set(sample_set)
        uniqueness = len(unique_smiles) / num_valid if num_valid else 0.0

        fps = _ecfp4_fingerprints(list(unique_smiles))
        diversity_ecfp4 = _mean_pairwise_tanimoto_distance(fps) if fps else 0.0

        ref_set = set(reference_smiles)
        novel = [s for s in unique_smiles if s not in ref_set]
        novelty = len(novel) / len(unique_smiles) if unique_smiles else 0.0

        qed_mean, qed_median, sa_mean, sa_median = qed_sa_distribution(list(unique_smiles))
        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "diversity": diversity_ecfp4,
            "diversity_ecfp4_mean_distance": diversity_ecfp4,
            "novelty": novelty,
            "qed_mean": qed_mean,
            "qed_median": qed_median,
            "sa_mean": sa_mean,
            "sa_median": sa_median,
        }


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
