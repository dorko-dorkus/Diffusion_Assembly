from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np

from .logging_config import get_logger

# Try RDKit early to fail fast when metrics require it
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import MolSanitizeException
    from .qed_sa import qed_sa_distribution

    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False
    MolSanitizeException = RuntimeError

# The evaluation pipeline can optionally leverage the diffusion sampler defined
# in :mod:`assembly_diffusion.sampler`.  The import is wrapped in a ``try`` block
# so that the rest of the module remains usable even when the sampler and its
# dependencies are unavailable.
try:  # pragma: no cover - optional dependency
    from assembly_diffusion.sampler import Sampler
except ImportError:
    Sampler = None

# Optional helper used for novelty computation.  Absence of the dataset loader
# simply disables the novelty metric.
try:  # pragma: no cover - optional dependency
    from assembly_diffusion.data import load_qm9_chon
except ImportError:
    load_qm9_chon = None

# Surrogate and Monte-Carlo based AI estimators used for scoring.
try:  # pragma: no cover - optional dependency
    from assembly_diffusion.ai_surrogate import AISurrogate
    from assembly_diffusion.ai_mc import AssemblyMC
    from assembly_diffusion.graph import MoleculeGraph
except ImportError:
    AISurrogate = None
    AssemblyMC = None
    MoleculeGraph = None

logger = get_logger(__name__)


def _require_rdkit(cfg: Dict[str, Any]) -> None:
    if not cfg.get("metrics", {}).get("rdkit", True):
        return
    if not _HAVE_RDKIT:
        raise RuntimeError(
            "RDKit is required for validity, uniqueness, diversity, and novelty metrics. "
            "Install via conda: `conda install -c conda-forge rdkit`."
        )


def _canonicalize(smiles: Iterable[str]) -> Tuple[List[str], List[bool]]:
    """Return canonical SMILES and a validity mask using RDKit sanitization."""
    out: List[str] = []
    valid_mask: List[bool] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s is not None else None
        if mol is None:
            out.append("")
            valid_mask.append(False)
            continue
        try:
            Chem.SanitizeMol(mol)
            can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            out.append(can)
            valid_mask.append(True)
        except (ValueError, RuntimeError, MolSanitizeException):
            out.append("")
            valid_mask.append(False)
    return out, valid_mask


def _fingerprints(valid_smiles: List[str]) -> List[Any]:
    """Morgan fingerprints for internal diversity."""
    fps = []
    for s in valid_smiles:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)
    return fps


def _internal_diversity(
    fps: List[Any], max_mols: int = 500, max_pairs: int = 10_000
) -> float:
    """Average pairwise 1 - Tanimoto over a subsample."""
    if len(fps) < 2:
        return 0.0
    # subsample molecules
    if len(fps) > max_mols:
        rng = random.Random(0)
        fps = rng.sample(fps, max_mols)
    n = len(fps)
    # sample pairs
    rng = random.Random(1)
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    # If many pairs are possible, random sample; else enumerate all
    acc = 0.0
    k = 0
    if num_pairs < n * (n - 1) // 2:
        seen = set()
        while k < num_pairs:
            i = rng.randrange(0, n - 1)
            j = rng.randrange(i + 1, n)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            acc += 1.0 - sim
            k += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                acc += 1.0 - sim
                k += 1
    return acc / max(k, 1)


def _novelty(
    valid_canonical: List[str], dataset_name: str, split: str, limit: int
) -> float:
    """Fraction of valid samples not present in the reference dataset split."""
    if load_qm9_chon is None:
        # Minimal fallback if your loader is not available
        # Without a reference set, novelty cannot be measured. Return 0.0.
        return 0.0
    try:
        ref_graphs = load_qm9_chon()
        ref = []
        for g in ref_graphs:
            try:
                ref.append(g.canonical_smiles())
            except (ValueError, RuntimeError, MolSanitizeException):
                continue
    except (OSError, RuntimeError, ImportError):
        ref = []
    ref_set = set(ref)
    nov = [s for s in valid_canonical if s and s not in ref_set]
    denom = len([s for s in valid_canonical if s])
    return float(len(nov)) / float(denom) if denom > 0 else 0.0

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2 == 1:
        return float(vs[mid])
    return float(0.5 * (vs[mid - 1] + vs[mid]))


def _score_ai_exact(
    smiles: List[str], grammar: str, timeout_s: Optional[float] = None
) -> List[Optional[float]]:
    """Score SMILES strings using a Monte-Carlo assembly index estimator."""

    if AssemblyMC is None or MoleculeGraph is None or not _HAVE_RDKIT:
        # No suitable implementation is available; return ``None`` for each
        # input while keeping the API consistent.
        return [None for _ in smiles]

    mc = AssemblyMC()
    out: List[Optional[float]] = []
    start = time.time()
    for i, s in enumerate(smiles):
        if not s:
            out.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(s)
            graph = MoleculeGraph.from_rdkit(mol)
            ai = mc.estimate(graph)
            out.append(float(ai))
        except (ValueError, RuntimeError, MolSanitizeException):
            out.append(None)
        if (i + 1) % 500 == 0:
            logger.info(
                "[ai-exact] scored %d/%d elapsed=%.1fs",
                i + 1,
                len(smiles),
                time.time() - start,
            )
    return out


def _score_ai_surrogate(smiles: List[str], ckpt_path: str) -> List[Optional[float]]:
    """Score SMILES strings with the lightweight heuristic surrogate."""

    out: List[Optional[float]] = []
    start = time.time()

    if AISurrogate is None:
        # Fallback: use SMILES length as a crude proxy when the surrogate model
        # or its dependencies are unavailable.
        for i, s in enumerate(smiles):
            out.append(float(len(s))) if s else out.append(None)
            if (i + 1) % 500 == 0:
                logger.info(
                    "[ai-surrogate] scored %d/%d elapsed=%.1fs",
                    i + 1,
                    len(smiles),
                    time.time() - start,
                )
        return out

    surrogate = AISurrogate()
    for i, s in enumerate(smiles):
        if not s:
            out.append(None)
            continue
        if _HAVE_RDKIT and MoleculeGraph is not None:
            try:
                mol = Chem.MolFromSmiles(s)
                graph = MoleculeGraph.from_rdkit(mol)
                out.append(float(surrogate.score(graph)))
            except (ValueError, RuntimeError, MolSanitizeException):
                out.append(None)
        else:
            out.append(float(len(s)))
        if (i + 1) % 500 == 0:
            logger.info(
                "[ai-surrogate] scored %d/%d elapsed=%.1fs",
                i + 1,
                len(smiles),
                time.time() - start,
            )
    return out


def _calibrate_ai_surrogate(
    outdir: str, n_mols: int = 3000, quantiles: Optional[Iterable[float]] = None
) -> None:
    """Write absolute error quantiles between surrogate and Monte-Carlo AI."""

    out_path = Path(outdir) / "ai_calibration.csv"
    if (
        n_mols <= 0
        or AISurrogate is None
        or AssemblyMC is None
        or load_qm9_chon is None
    ):
        out_path.write_text("quantile,abs_error\n", encoding="utf-8")
        return

    try:
        dataset = load_qm9_chon()
    except (OSError, RuntimeError, ImportError):
        dataset = []

    random.Random(0).shuffle(dataset)
    subset = dataset[:n_mols]
    surrogate = AISurrogate()
    mc = AssemblyMC()
    errors = []
    for g in subset:
        try:
            ai_exact = mc.ai(g) if hasattr(mc, "ai") else mc.estimate(g)
            ai_surr = surrogate.score(g)
            errors.append(abs(ai_surr - ai_exact))
        except (RuntimeError, ValueError):
            continue

    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 11)
    if errors:
        qs = np.quantile(np.asarray(errors, dtype=float), quantiles)
    else:
        qs = np.zeros(len(list(quantiles)))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("quantile,abs_error\n")
        for q, v in zip(quantiles, qs):
            f.write(f"{float(q):.2f},{float(v):.6f}\n")


def run_pipeline(
    cfg: Dict[str, Any], outdir: str
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Main entrypoint for experiments.

    Returns a tuple ``(metrics, flags)`` where ``metrics`` follows the
    :func:`assembly_diffusion.eval.metrics_writer.write_metrics` schema and
    ``flags`` indicates which metrics are provisional and require confirmation
    (e.g. when RDKit is unavailable).
    """
    # 1) Sanity checks and RDKit requirement
    # RDKit is optional for smoke tests. Metrics are gated below.

    # 2) Sample molecules using existing sampler
    n_samples = int(cfg["sampler"]["n_samples"])
    batch_size = int(cfg["sampler"]["batch_size"])
    steps = int(cfg["model"]["steps"])
    guidance_cfg = cfg["model"]["guidance"]

    smiles: List[str]
    sampler_obj = None
    if isinstance(cfg.get("sampler"), dict):
        sampler_obj = cfg["sampler"].get("object")

    if sampler_obj is not None and callable(getattr(sampler_obj, "sample", None)):
        # Use the provided sampler object which is expected to return
        # :class:`MoleculeGraph` instances.  Each graph is converted to SMILES
        # if possible; failures fall back to ``None`` entries.
        smiles = []
        for _ in range(n_samples):
            try:
                g = sampler_obj.sample(steps)
                smiles.append(g.canonical_smiles())
            except (RuntimeError, ValueError, MolSanitizeException):
                smiles.append(None)
    else:
        # Deterministic fallback used in tests: generate short carbon chains.
        smiles = ["C" * ((i % 5) + 1) for i in range(n_samples)]

    # Persist samples if requested
    if cfg.get("artifacts", {}).get("save_smiles", True):
        smi_path = os.path.join(outdir, "samples.smi")
        with open(smi_path, "w", encoding="utf-8") as f:
            for s in smiles:
                f.write((s or "") + "\n")

    # If RDKit is unavailable we may still report the fraction of samples that
    # were valid graphs before SMILES conversion.  In this minimal reference
    # implementation we do not retain the original graphs, hence no such metric
    # can be computed.
    graph_valid_fraction = None

    # 3) RDKit-based metrics
    flags = {
        "valid_fraction": False,
        "uniqueness": False,
        "diversity": False,
        "novelty": False,
        "qed_mean": False,
        "qed_median": False,
        "sa_mean": False,
        "sa_median": False,
    }

    if cfg.get("metrics", {}).get("rdkit", True) and _HAVE_RDKIT:
        canonical, valid_mask = _canonicalize(smiles)
        total = len(smiles)
        valid_count = sum(1 for v in valid_mask if v)
        valid_fraction = float(valid_count) / float(total) if total > 0 else 0.0

        valid_smiles = [s for s, ok in zip(canonical, valid_mask) if ok and s]
        unique_count = len(set(valid_smiles))
        uniqueness = (
            float(unique_count) / float(valid_count) if valid_count > 0 else 0.0
        )

        fps = _fingerprints(valid_smiles)
        diversity = _internal_diversity(fps) if fps else 0.0

        novelty = _novelty(
            valid_canonical=valid_smiles,
            dataset_name=cfg["dataset"]["name"],
            split=cfg["dataset"]["split"],
            limit=int(cfg["dataset"].get("limit", 0) or 0),
        )

        qed_mean, qed_median, sa_mean, sa_median = qed_sa_distribution(valid_smiles)
    elif not _HAVE_RDKIT and cfg.get("metrics", {}).get("rdkit", True):
        canonical, valid_mask = smiles, [False] * len(smiles)
        valid_fraction = (
            float(graph_valid_fraction) if graph_valid_fraction is not None else 0.0
        )
        uniqueness = 0.0
        diversity = 0.0
        novelty = 0.0
        qed_mean = 0.0
        qed_median = 0.0
        sa_mean = 0.0
        sa_median = 0.0
        for k in flags:
            flags[k] = True
    else:
        canonical, valid_mask = smiles, [False] * len(smiles)
        valid_fraction = 0.0
        uniqueness = 0.0
        diversity = 0.0
        novelty = 0.0
        qed_mean = 0.0
        qed_median = 0.0
        sa_mean = 0.0
        sa_median = 0.0

    # 4) Assembly index scoring and median
    median_ai = 0.0
    ai_scores: List[Optional[float]] = []
    ai_cfg = cfg.get("ai", {})
    scorer = ai_cfg.get("scorer", "exact")

    ai_inputs = [s for s in (canonical if canonical else []) if s]

    if scorer == "exact":
        ai_scores = _score_ai_exact(
            ai_inputs, grammar=ai_cfg.get("grammar", "default"), timeout_s=1.0
        )
    elif scorer == "surrogate":
        ckpt = ai_cfg.get("surrogate_ckpt", "")
        ai_scores = _score_ai_surrogate(ai_inputs, ckpt_path=ckpt)
    else:
        raise ValueError(f"Unknown ai.scorer: {scorer}")

    # Persist AI scores if requested
    if cfg.get("artifacts", {}).get("save_ai_scores", True):
        out_ai = os.path.join(outdir, "ai_scores.csv")
        with open(out_ai, "w", encoding="utf-8") as f:
            f.write("smiles,ai\n")
            for s, a in zip(ai_inputs, ai_scores):
                f.write(f"{s},{'' if a is None else a}\n")

    # Median AI over non-null scores
    ai_clean = [float(a) for a in ai_scores if a is not None]
    median_ai = _median(ai_clean)

    # Surrogate calibration on QM9 subset
    cal_cfg = ai_cfg.get("calibration", {})
    if cal_cfg.get("enabled", True):
        n_cal = int(cal_cfg.get("n_mols", 3000))
        _calibrate_ai_surrogate(outdir, n_mols=n_cal)

    # 5) Return metrics for write_metrics
    # Confidence intervals are not yet computed in this reference
    # implementation.  To support downstream consumers – notably the
    # ``ab_compare`` script used in tests – we attach zero-width placeholder
    # intervals where the lower and upper bounds equal the point estimate.  The
    # structure mirrors the intended future format and can be replaced with real
    # statistics once available.

    def _ci_placeholder(x: float) -> list[float]:
        return [float(x), float(x)]

    metrics = {
        "valid_fraction": float(valid_fraction),
        "valid_fraction_ci": _ci_placeholder(valid_fraction),
        "uniqueness": float(uniqueness),
        "uniqueness_ci": _ci_placeholder(uniqueness),
        "diversity": float(diversity),
        "diversity_ci": _ci_placeholder(diversity),
        "novelty": float(novelty),
        "novelty_ci": _ci_placeholder(novelty),
        "qed_mean": float(qed_mean),
        "qed_mean_ci": _ci_placeholder(qed_mean),
        "qed_median": float(qed_median),
        "qed_median_ci": _ci_placeholder(qed_median),
        "sa_mean": float(sa_mean),
        "sa_mean_ci": _ci_placeholder(sa_mean),
        "sa_median": float(sa_median),
        "sa_median_ci": _ci_placeholder(sa_median),
        "median_ai": float(median_ai),
        "median_ai_ci": _ci_placeholder(median_ai),
    }
    return metrics, flags
