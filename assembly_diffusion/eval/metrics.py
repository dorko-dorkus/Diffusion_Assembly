"""Experiment specification for RDKit-based evaluation metrics.

baseline: RDKit validity, uniqueness, diversity and novelty metrics provide the
    baseline quality assessment. ``evaluate_with_baselines`` enables control runs
    to compare generated molecules against simple references or ablations.
data_sources: generated molecule sets and reference SMILES such as the QM9-CHON
    training split.
method: canonicalise molecules, compute ECFP4 fingerprints and Tanimoto
    distances, and summarise QED and SA scores with RDKit.
objective: quantify quality and diversity of generated molecules for comparison
    to baselines.
params: fingerprint radius=2 and ``nBits``=2048; evaluate ``sample_set`` against
    ``reference_smiles``.
repro: deterministic RDKit operations ensure reproducible metrics for fixed
    inputs.
validation: generated molecules can be split into train/validation/test sets or
    used in ``k``-fold cross-validation for model selection. Early stopping
    should monitor a validation metric such as ``qed_mean`` with a patience of
    ``n`` epochs. ``tests/test_metrics_rdkit_required.py`` checks dependency
    handling and expected metric keys.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.1.0"

import random
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Sequence, Set, List, Any, Tuple, Optional, Mapping, Dict
from importlib import metadata

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from ..qed_sa import qed_sa_distribution
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    DataStructs = None
    AllChem = None
    qed_sa_distribution = None

from ..graph import MoleculeGraph
from .validity import sanitize_or_none
from ..ai_surrogate import AISurrogate
from ..logging_config import get_logger


logger = get_logger(__name__)


def _git_hash() -> str:
    """Return the current git commit hash or ``"unknown"``."""

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return "unknown"
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=repo_root,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _configure_reproducibility(seed: int) -> None:
    """Set random seeds and log environment and version information."""

    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU optional
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass

    logger.info("Random seed set to %d", seed)
    logger.info("Python %s on %s", platform.python_version(), platform.platform())

    for pkg in ("rdkit", "numpy", "torch"):
        try:
            ver = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            ver = "not installed"
        logger.info("%s version: %s", pkg, ver)

    logger.info("Git commit SHA: %s", _git_hash())



def smiles_set(graphs: Iterable[MoleculeGraph]) -> Set[str]:
    """Return a set of canonical SMILES for valid graphs."""

    if Chem is None:
        raise RuntimeError("RDKit required for metric smiles_set")
    result: Set[str] = set()
    for g in graphs:
        mol = sanitize_or_none(g)
        if mol is not None:
            result.add(Chem.MolToSmiles(mol, canonical=True))
    return result


def _ecfp4_fingerprints(valid_smiles: List[str]) -> List[Any]:
    """Return RDKit ECFP4 fingerprints for ``valid_smiles``."""

    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit required for ECFP4 fingerprints")

    fps: List[Any] = []
    for s in valid_smiles:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def _mean_pairwise_tanimoto_distance(fps: List[Any]) -> float:
    """Return the mean pairwise ``1 - Tanimoto`` distance for fingerprints."""

    if DataStructs is None:
        raise RuntimeError("RDKit required for Tanimoto similarity")

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


def train_val_test_split(
    graphs: Sequence[MoleculeGraph],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[MoleculeGraph], List[MoleculeGraph], List[MoleculeGraph]]:
    """Return shuffled ``graphs`` split into train, validation and test sets.

    The ratios must sum to 1.0. ``seed`` controls the RNG used for shuffling.
    This helper enables a simple validation protocol when evaluating models.
    """

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    items = list(graphs)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


class EarlyStopping:
    """Track validation scores and implement early stopping.

    ``patience`` specifies the number of epochs without improvement allowed
    before signalling that training should stop. ``mode`` determines whether a
    higher (``"max"``) or lower (``"min"``) score is better.
    """

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Update with a new ``metric``. Return ``True`` to stop training."""

        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Metrics:
    """Basic metrics for evaluating generated molecules."""

    @staticmethod
    def evaluate(
        sample_set: Sequence[MoleculeGraph],
        reference_smiles: Iterable[str],
        seed: int = 0,
    ) -> dict[str, float]:
        """Return RDKit-based metrics for ``sample_set``.

        The returned dictionary contains ``validity``, ``uniqueness``,
        ``diversity`` (legacy alias for ``diversity_ecfp4_mean_distance``),
        ``diversity_ecfp4_mean_distance`` (mean pairwise 1 - Tanimoto distance
        over ECFP4 fingerprints), ``novelty`` against the provided
        ``reference_smiles``, and distribution summaries for RDKit ``qed``
        and synthetic accessibility ``sa`` scores.
        seed: int, optional
            Random seed used for deterministic operations and environment
            logging. Defaults to ``0``.
        """

        if Chem is None:
            raise RuntimeError("RDKit required for metric evaluate")

        _configure_reproducibility(seed)
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

    @staticmethod
    def evaluate_with_baselines(
        sample_set: Sequence[MoleculeGraph],
        reference_smiles: Iterable[str],
        baselines: Optional[Mapping[str, Sequence[MoleculeGraph]]] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate ``sample_set`` alongside baseline or control molecule sets.

        Parameters
        ----------
        sample_set:
            Molecules generated by the model under evaluation.
        reference_smiles:
            Reference SMILES used when computing novelty.
        baselines:
            Optional mapping from a baseline name to a sequence of molecules.
            Each baseline is scored with the same metrics to provide a simple
            control or ablation comparison.
        seed:
            Random seed forwarded to :meth:`evaluate` for deterministic
            operations.

        Returns
        -------
        Dict[str, Any]
            Dictionary with a ``sample`` entry containing the metrics for
            ``sample_set`` and, when ``baselines`` is provided, a ``baselines``
            mapping with metric dictionaries for each baseline.
        """

        result: Dict[str, Any] = {
            "sample": Metrics.evaluate(sample_set, reference_smiles, seed)
        }
        if baselines:
            baseline_metrics: Dict[str, Any] = {}
            for name, graphs in baselines.items():
                baseline_metrics[name] = Metrics.evaluate(graphs, reference_smiles, seed)
            result["baselines"] = baseline_metrics
        return result


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
