from __future__ import annotations
import argparse
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from math import log
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=str)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--boot", type=int, default=1000)


def fit_logfreq_vs_A(df: pd.DataFrame, trim_quant: float = 0.05) -> Tuple[float, float, float]:
    df = df.copy()
    df["A"] = df["As_upper"].astype(int)
    grouped = df.groupby("A")["frequency"].sum().reset_index()
    lo, hi = grouped["A"].quantile([trim_quant, 1 - trim_quant])
    mask = (grouped["A"] >= lo) & (grouped["A"] <= hi)
    trimmed = grouped[mask]
    X = trimmed["A"].to_numpy()
    y = np.log(np.maximum(trimmed["frequency"].to_numpy(), 1e-12))
    A = np.vstack([X, np.ones_like(X)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m * X + c
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(c), float(r2)


def bootstrap_ci(df: pd.DataFrame, alpha: float = 0.05, B: int = 1000) -> Tuple[float, float]:
    m_vals = []
    for _ in range(B):
        resampled = df.sample(n=len(df), replace=True)
        try:
            m, c, r2 = fit_logfreq_vs_A(resampled)
            m_vals.append(m)
        except (ValueError, RuntimeError):
            continue
    m_vals = np.array(m_vals)
    lo = np.quantile(m_vals, alpha/2)
    hi = np.quantile(m_vals, 1 - alpha/2)
    return float(lo), float(hi)


def degeneracy_spearman(df: pd.DataFrame) -> Tuple[float, float]:
    sub = df.dropna(subset=["d_min"]).copy()
    if sub.empty:
        return np.nan, np.nan
    sub["A"] = sub["As_upper"].astype(int)
    rhos = []
    ps = []
    for A, g in sub.groupby("A"):
        if g["d_min"].nunique() > 1 and g.shape[0] >= 4:
            rho, p = spearmanr(g["frequency"], g["d_min"])
            rhos.append(rho); ps.append(p)
    if not rhos:
        return np.nan, np.nan
    return float(np.nanmean(rhos)), float(np.nanmean(ps))

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    m, c, r2 = fit_logfreq_vs_A(df)
    m_lo, m_hi = bootstrap_ci(df, alpha=args.alpha, B=args.boot)
    rho, p = degeneracy_spearman(df)
    logger.info(
        "%s",
        {
            "m": m,
            "m_CI": [m_lo, m_hi],
            "R2": r2,
            "rho_deg": rho,
            "p_deg": p,
            "N_rows": int(df.shape[0]),
        },
    )
