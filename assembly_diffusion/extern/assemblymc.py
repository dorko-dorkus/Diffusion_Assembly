"""Adapter for calling the external AssemblyMC binary.

The wrapper resolves the executable path from the ``ASSEMBLYMC_BIN``
environment variable (preferred) or falls back to a configured
``cfg.ai.bin_path`` if available.  Results are cached both in memory (LRU)
and on disk keyed by the molecule InChIKey.  The external program is
expected to print ``A_star`` and ``d_min_est`` on stdout and write a
``stats.json`` file in its working directory.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, Tuple, Optional

try:  # pragma: no cover - exercised when RDKit is absent
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

__all__ = [
    "AssemblyMCError",
    "AssemblyMCTimeout",
    "a_star_and_dmin",
]


class AssemblyMCError(RuntimeError):
    """Base class for AssemblyMC adapter errors."""


class AssemblyMCTimeout(AssemblyMCError):
    """Raised when the external binary times out."""


_CACHE_MAX = 32
_CACHE: "OrderedDict[str, Tuple[int, Optional[int], Dict]]" = OrderedDict()


def _require_rdkit() -> None:  # pragma: no cover - trivial
    if Chem is None:
        raise ImportError("RDKit is required for AssemblyMC operations")


def _cache_dir() -> Path:
    path = os.environ.get("ASSEMBLYMC_CACHE")
    if path:
        cd = Path(path)
    else:
        cd = Path.home() / ".cache" / "assemblymc"
    cd.mkdir(parents=True, exist_ok=True)
    return cd


def _cache_get(key: str) -> Optional[Tuple[int, Optional[int], Dict]]:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    path = _cache_dir() / f"{key}.json"
    if path.exists():
        with path.open("r", encoding="utf8") as f:
            data = json.load(f)
        res = (
            int(data.get("A_star", 0)),
            data.get("d_min_est"),
            data.get("stats", {}),
        )
        _CACHE[key] = res
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
        return res
    return None


def _cache_put(key: str, value: Tuple[int, Optional[int], Dict]) -> None:
    _CACHE[key] = value
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    path = _cache_dir() / f"{key}.json"
    with path.open("w", encoding="utf8") as f:
        json.dump({"A_star": value[0], "d_min_est": value[1], "stats": value[2]}, f)


def _resolve_bin() -> Path:
    env = os.environ.get("ASSEMBLYMC_BIN")
    if env:
        return Path(env)
    # Fallback to a configured path if available
    try:  # pragma: no cover - config may not exist in tests
        from .. import _config  # type: ignore  # hypothetical runtime config

        cfg_bin = getattr(getattr(_config, "ai", object()), "bin_path", None)
        if cfg_bin:
            return Path(cfg_bin)
    except Exception:
        pass
    raise AssemblyMCError(
        "AssemblyMC binary not configured. Obtain permission, compile the binary, "
        "and set ASSEMBLYMC_BIN or cfg.ai.bin_path."
    )


def a_star_and_dmin(
    smiles: str, trials: int = 2000, seed: int = 0, timeout_s: float = 2.0
) -> Tuple[int, Optional[int], Dict]:
    """Estimate Assembly index and minimal D via external solver.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.
    trials:
        Monte-Carlo trials passed to the external binary.
    seed:
        RNG seed for the binary.
    timeout_s:
        Maximum runtime in seconds before aborting the subprocess.
    """

    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    key = Chem.MolToInchiKey(mol)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    bin_path = _resolve_bin()

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [str(bin_path), smiles, str(trials), str(seed)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised in tests
            raise AssemblyMCTimeout(f"AssemblyMC timed out after {timeout_s} s") from exc

        out = proc.stdout.splitlines()
        a_star: Optional[int] = None
        d_min: Optional[int] = None
        for line in out:
            if line.lower().startswith("a_star"):
                try:
                    a_star = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            if line.lower().startswith("d_min"):
                try:
                    d_min = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        stats_path = Path(tmpdir) / "stats.json"
        stats: Dict = {}
        if stats_path.exists():
            with stats_path.open("r", encoding="utf8") as f:
                stats = json.load(f)

    if a_star is None:
        raise AssemblyMCError("AssemblyMC output did not contain 'A_star'")

    result = (a_star, d_min, stats)
    _cache_put(key, result)
    return result
