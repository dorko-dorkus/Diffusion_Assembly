from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


def _git_hash() -> str:
    """Return the current git commit hash or ``"unknown"``."""
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        return "unknown"
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def setup_run_logger(
    log_file: str | Path,
    grammar: str,
    config: Dict[str, Any],
    seeds: Dict[str, int] | None = None,
) -> logging.Logger:
    """Configure a file logger that writes a JSON header.

    Parameters
    ----------
    log_file:
        Destination log file. Parent directories are created automatically.
    grammar:
        Grammar label for the run (e.g. ``"G"`` or ``"G_MC"``).
    config:
        Configuration object to record in the header.
    seeds:
        Optional mapping of RNG names to the seeds used.

    Returns
    -------
    logging.Logger
        Logger instance writing to ``log_file``.

    The first line of ``log_file`` is a JSON object containing ``seeds``,
    ``package_versions``, ``commit``, ``grammar`` and ``config`` keys. Subsequent
    log records are appended in human readable format.
    """

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "pandas", "scipy"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            # Package not installed or version unavailable; skip.
            continue

    header = {
        "seeds": seeds or {},
        "package_versions": versions,
        "commit": _git_hash(),
        "grammar": grammar,
        "config": config,
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False)
        f.write("\n")

    logger = logging.getLogger("assembly_diffusion.run")
    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


__all__ = ["setup_run_logger"]
