from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import platform
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

_RUN_LOGGER = logging.getLogger("assembly_diffusion.run")


def _git_hash() -> str:
    """Return the current git commit hash or "unknown"."""
    repo_root = Path(__file__).resolve().parents[1]
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


def _package_versions() -> Dict[str, str]:
    """Best-effort collection of package versions."""
    versions: Dict[str, str] = {}
    for pkg in ("assembly_diffusion", "numpy", "torch", "rdkit"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    versions["python"] = sys.version.split()[0]
    return versions


def _set_seeds(seed: int) -> Dict[str, int]:
    seeds: Dict[str, int] = {}
    random.seed(seed)
    seeds["python"] = seed
    try:
        import numpy as np

        np.random.seed(seed)
        seeds["numpy"] = seed
    except ImportError:
        pass
    try:  # Optional torch seed
        import torch  # type: ignore

        torch.manual_seed(seed)
        seeds["torch"] = seed
    except ImportError:
        pass
    return seeds


def reset_run_logger() -> None:
    """Remove all handlers from the module logger.

    This is primarily intended for use in tests where a fresh logger is
    required.  It closes any existing handlers to release open file
    descriptors.
    """

    for handler in list(_RUN_LOGGER.handlers):
        _RUN_LOGGER.removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass


def init_run_logger(
    log_path: str,
    grammar: str,
    config: dict,
    seed: Optional[int] = None,
    grammar_text: Optional[str] = None,
) -> logging.Logger:
    """Initialise a file logger that writes a JSON header.

    The header contains RNG seeds, package versions, git commit hash,
    operating system details, the command line, the grammar label, and the
    configuration blob.  It is written to the log file before any other
    records.
    """

    if _RUN_LOGGER.handlers:
        return _RUN_LOGGER

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    seeds = _set_seeds(seed) if seed is not None else {}
    header = {
        "seeds": seeds,
        "packages": _package_versions(),
        "git_hash": _git_hash(),
        "os_version": platform.platform(),
        "command": " ".join(sys.argv),
        "grammar": grammar,
        "grammar_text": grammar_text,
        "config": config,
    }
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(header, fh)
        fh.write("\n")

    handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    _RUN_LOGGER.addHandler(handler)
    _RUN_LOGGER.setLevel(logging.INFO)
    return _RUN_LOGGER


__all__ = ["init_run_logger", "reset_run_logger"]
