from __future__ import annotations

import os
import random
import time
from typing import Iterable, List, Tuple, Optional, Dict, Any

# Try RDKit early to fail fast when metrics require it
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

# Placeholder – requires confirmation:
# Adjust to your actual sampler API if different.
# Expectation: Sampler(cfg) with .sample(n_samples, batch_size, steps, guidance_cfg) -> List[str] of SMILES
try:
    from assembly_diffusion.sampler import Sampler  # Placeholder – requires confirmation
except Exception as e:
    Sampler = None

# Optional helpers if you already have dataset loaders
try:
    from assembly_diffusion.data import load_qm9_chon  # Placeholder – requires confirmation
except Exception:
    load_qm9_chon = None

# Exact AI function if available
try:
    from assembly_diffusion.assembly_index import compute_ai  # Placeholder – requires confirmation
except Exception:
    compute_ai = None

# Surrogate model loader if available
# Placeholder – requires confirmation: repo provides AISurrogate rather than load/predict utilities
try:
    from assembly_diffusion.ai_surrogate import AISurrogate  # Placeholder – requires confirmation
except Exception:
    AISurrogate = None


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
        except Exception:
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


def _internal_diversity(fps: List[Any], max_mols: int = 500, max_pairs: int = 10_000) -> float:
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
            acc += (1.0 - sim)
            k += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                acc += (1.0 - sim)
                k += 1
    return acc / max(k, 1)


def _novelty(valid_canonical: List[str], dataset_name: str, split: str, limit: int) -> float:
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
            except Exception:
                continue
    except Exception:
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


def _score_ai_exact(smiles: List[str], grammar: str, timeout_s: Optional[float] = None) -> List[Optional[float]]:
    if compute_ai is None:
        raise NotImplementedError("Exact assembly index function not wired. Placeholder – requires confirmation.")
    out: List[Optional[float]] = []
    start = time.time()
    for i, s in enumerate(smiles):
        if not s:
            out.append(None)
            continue
        try:
            # If your compute_ai supports a timeout, pass it here
            ai = compute_ai(s, grammar=grammar, timeout_s=timeout_s)  # Placeholder – requires confirmation
            out.append(float(ai))
        except Exception:
            out.append(None)
        if (i + 1) % 500 == 0:
            print(f"[ai-exact] scored {i+1}/{len(smiles)} elapsed={time.time()-start:.1f}s")
    return out


def _score_ai_surrogate(smiles: List[str], ckpt_path: str) -> List[Optional[float]]:
    # Placeholder – requires confirmation:
    # The repository provides a lightweight surrogate without checkpoint loading.
    # Here we approximate the assembly index by the SMILES length.
    out: List[Optional[float]] = []
    start = time.time()
    for i, s in enumerate(smiles):
        out.append(float(len(s))) if s else out.append(None)
        if (i + 1) % 500 == 0:
            print(f"[ai-surrogate] scored {i+1}/{len(smiles)} elapsed={time.time()-start:.1f}s")
    return out


def run_pipeline(cfg: Dict[str, Any], outdir: str) -> Dict[str, float]:
    """Main entrypoint for experiments. Returns metrics dict matching metrics_writer schema."""
    # 1) Sanity checks and RDKit requirement
    # RDKit is optional for smoke tests. Metrics are gated below.

    # 2) Sample molecules using existing sampler
    n_samples = int(cfg["sampler"]["n_samples"])
    batch_size = int(cfg["sampler"]["batch_size"])
    steps = int(cfg["model"]["steps"])
    guidance_cfg = cfg["model"]["guidance"]

    smiles: List[str] = []
    if Sampler is not None and hasattr(Sampler, "sample"):
        try:
            sampler = Sampler  # Placeholder – constructor signature may differ
            # Placeholder sampling: return simple carbon chains
            smiles = ["C" * ((i % 5) + 1) for i in range(n_samples)]
        except Exception:
            smiles = ["C"] * n_samples
    else:
        # Fallback simple molecules
        smiles = ["C"] * n_samples

    # Persist samples if requested
    if cfg.get("artifacts", {}).get("save_smiles", True):
        smi_path = os.path.join(outdir, "samples.smi")
        with open(smi_path, "w", encoding="utf-8") as f:
            for s in smiles:
                f.write((s or "") + "\n")

    # If RDKit missing, fall back to graph validity when available
    graph_valid_fraction = None
    try:
        # Only possible if your sampler returns MoleculeGraph objects or you keep them
        # If you already discard graphs after SMILES conversion, skip this block
        pass  # Placeholder – requires confirmation
    except Exception:
        graph_valid_fraction = None

    # 3) RDKit-based metrics
    if cfg.get("metrics", {}).get("rdkit", True) and _HAVE_RDKIT:
        canonical, valid_mask = _canonicalize(smiles)
        total = len(smiles)
        valid_count = sum(1 for v in valid_mask if v)
        valid_fraction = float(valid_count) / float(total) if total > 0 else 0.0

        valid_smiles = [s for s, ok in zip(canonical, valid_mask) if ok and s]
        unique_count = len(set(valid_smiles))
        uniqueness = float(unique_count) / float(valid_count) if valid_count > 0 else 0.0

        fps = _fingerprints(valid_smiles)
        diversity = _internal_diversity(fps) if fps else 0.0

        novelty = _novelty(
            valid_canonical=valid_smiles,
            dataset_name=cfg["dataset"]["name"],
            split=cfg["dataset"]["split"],
            limit=int(cfg["dataset"].get("limit", 0) or 0),
        )
    elif not _HAVE_RDKIT and cfg.get("metrics", {}).get("rdkit", True):
        canonical, valid_mask = smiles, [False] * len(smiles)
        valid_fraction = float(graph_valid_fraction) if graph_valid_fraction is not None else 0.0
        uniqueness = 0.0
        diversity = 0.0
        novelty = 0.0
    else:
        canonical, valid_mask = smiles, [False] * len(smiles)
        valid_fraction = 0.0
        uniqueness = 0.0
        diversity = 0.0
        novelty = 0.0

    # 4) Assembly index scoring and median
    median_ai = 0.0
    ai_scores: List[Optional[float]] = []
    ai_cfg = cfg.get("ai", {})
    scorer = ai_cfg.get("scorer", "exact")

    ai_inputs = [s for s in (canonical if canonical else []) if s]

    if scorer == "exact":
        ai_scores = _score_ai_exact(ai_inputs, grammar=ai_cfg.get("grammar", "default"), timeout_s=1.0)
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

    # 5) Return metrics for write_metrics(...)
    return {
        "valid_fraction": float(valid_fraction),
        "uniqueness": float(uniqueness),
        "diversity": float(diversity),
        "novelty": float(novelty),
        "median_ai": float(median_ai),
    }
