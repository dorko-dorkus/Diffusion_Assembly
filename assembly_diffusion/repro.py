"""Utilities for reproducible experiments."""

from __future__ import annotations

import logging
import platform
import random
import subprocess

import numpy as np

try:  # Optional dependencies; versions logged if installed
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be absent
    pd = None  # type: ignore

try:
    import scipy
except Exception:  # pragma: no cover - scipy may be absent
    scipy = None  # type: ignore

try:
    import statsmodels
except Exception:  # pragma: no cover - statsmodels may be absent
    statsmodels = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = None  # type: ignore


def setup_reproducibility(seed: int = 0) -> None:
    """Seed RNGs and log environment information.

    Parameters
    ----------
    seed:
        Random seed used to initialise ``random``, ``numpy`` and, if available,
        ``torch``.  Environment details and selected package versions are logged
        alongside the current Git commit SHA.
    """

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git may be unavailable
        commit = "unknown"

    logger = logging.getLogger(__name__)
    logger.info(
        "Reproducibility: seed=%s python=%s numpy=%s pandas=%s scipy=%s statsmodels=%s torch=%s commit=%s",
        seed,
        platform.python_version(),
        np.__version__,
        pd.__version__ if pd is not None else "N/A",
        scipy.__version__ if scipy is not None else "N/A",
        statsmodels.__version__ if statsmodels is not None else "N/A",
        torch.__version__ if torch is not None else "N/A",
        commit,
    )
