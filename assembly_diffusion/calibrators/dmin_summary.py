from __future__ import annotations

"""Utilities for summarizing d_min estimates from calibrator runs."""

from typing import Tuple

import pandas as pd
from scipy.stats import spearmanr


def summarize_dmin(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Return per-``A`` Spearman stats and fraction of non-null ``d_min``.

    Parameters
    ----------
    df:
        DataFrame with at least ``As_upper``, ``d_min`` and ``frequency`` columns.

    Returns
    -------
    summary, frac
        ``summary`` contains ``A``, Spearman ``rho``/``p`` and group sizes ``n``.
        ``frac`` is the fraction of rows where ``d_min`` is not null.
    """

    frac = float(df["d_min"].notna().mean())
    sub = df.dropna(subset=["d_min"]).copy()
    sub["A"] = sub["As_upper"].astype(int)
    rows = []
    for A, g in sub.groupby("A"):
        if g["d_min"].nunique() > 1 and g.shape[0] >= 4:
            rho, p = spearmanr(g["frequency"], g["d_min"])
            rows.append({"A": int(A), "rho": float(rho), "p": float(p), "n": int(g.shape[0])})
    summary = pd.DataFrame(rows).sort_values("A").reset_index(drop=True)
    return summary, frac
