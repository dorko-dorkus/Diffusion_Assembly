"""Statistical helpers for assembly index summaries."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np


def summarise_A_hat(values: Sequence[Optional[float]]) -> dict[str, float]:
    """Return robust summary statistics for assembly index estimates.

    Parameters
    ----------
    values:
        Sequence of assembly index estimates which may contain ``None`` or
        ``nan`` entries.

    Returns
    -------
    dict[str, float]
        Dictionary with ``A_hat_median``, ``A_hat_IQR``, ``A_hat_p10`` and
        ``A_hat_p90`` fields.
    """

    arr = np.array([
        np.nan if v is None else float(v) for v in values
    ], dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return {
            "A_hat_median": 0.0,
            "A_hat_IQR": 0.0,
            "A_hat_p10": 0.0,
            "A_hat_p90": 0.0,
        }

    median = float(np.nanmedian(arr))
    p10, p90 = np.nanpercentile(arr, [10.0, 90.0])
    q1, q3 = np.nanpercentile(arr, [25.0, 75.0])
    iqr = float(q3 - q1)
    return {
        "A_hat_median": median,
        "A_hat_IQR": iqr,
        "A_hat_p10": float(p10),
        "A_hat_p90": float(p90),
    }


__all__ = ["summarise_A_hat"]
