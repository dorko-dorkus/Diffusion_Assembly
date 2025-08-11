"""Experiment specification for analyzing calibrator outputs.

baseline: frequency tables produced by ``scripts/run_calibrators.py`` serve as
    the reference distributions.
control: passing ``--baseline`` shuffles frequencies to break structure,
    providing a simple control run for comparison.
method: load a calibrator CSV, fit a trimmed log–linear model of frequency
    versus ``A`` and bootstrap slope estimates; measure Spearman correlation of
    degeneracy against frequency.
metrics: slope ``m``, intercept ``c``, coefficient of determination ``R²``,
    bootstrap confidence interval for ``m`` and Spearman ``ρ``/``p`` for
    degeneracy.  ``m`` and ``c`` are obtained from the least–squares fit of
    ``y = \log(f)`` to ``y = mA + c`` where
    ``m, c = \arg\min_{m,c} \sum_i (y_i - (m A_i + c))^2``.
    The coefficient of determination is
    ``R² = 1 - SS_res/SS_tot`` with
    ``SS_res = \sum_i (y_i - \hat y_i)^2`` and
    ``SS_tot = \sum_i (y_i - \bar y)^2``.
    Spearman rank correlation is computed as
    ``ρ = 1 - (6\sum_i d_i^2)/(n(n^2-1))`` with ``p`` from a ``t``-approximation.
reporting: metrics are averaged over random seeds and summarized as
    ``mean ± std``; for ``m`` the bootstrap distribution additionally provides
    a ``(1-α)`` confidence interval.
objective: quantify how calibration frequency scales with assembly size and how
    degeneracy relates to frequency.
repro: ``python scripts/analyze_calibrators.py path/to/calibs.csv --alpha 0.05
    --boot 1000`` reproduces the reported statistics.
validation: ``pytest tests/test_calibrators.py`` and
    ``pytest tests/test_analysis.py`` exercise the samplers and statistical
    helpers used here.  Optional cross-validation (``--cv``) selects the model
    with the highest validation ``R²`` and early stopping (``--patience``)
    halts evaluation when this metric fails to improve for consecutive folds.
"""

from __future__ import annotations

import argparse
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import early_stopping
from assembly_diffusion.repro import setup_reproducibility

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=str)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--cv", type=int, default=1, help="number of cross-validation folds")
parser.add_argument("--patience", type=int, default=0, help="early stopping patience in folds")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
parser.add_argument(
    "--baseline",
    action="store_true",
    help="include control run with shuffled frequencies",
)


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
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("Bootstrap iteration skipped: %s", exc)
            continue
    m_vals = np.array(m_vals)
    lo = np.quantile(m_vals, alpha/2)
    hi = np.quantile(m_vals, 1 - alpha/2)
    return float(lo), float(hi)


def eval_logfreq_vs_A(
    df: pd.DataFrame, m: float, c: float, trim_quant: float = 0.05
) -> float:
    """Evaluate ``R²`` of a pre-fit model ``y = mA + c`` on ``df``."""

    df = df.copy()
    df["A"] = df["As_upper"].astype(int)
    grouped = df.groupby("A")["frequency"].sum().reset_index()
    lo, hi = grouped["A"].quantile([trim_quant, 1 - trim_quant])
    mask = (grouped["A"] >= lo) & (grouped["A"] <= hi)
    trimmed = grouped[mask]
    if trimmed.empty:
        return np.nan
    X = trimmed["A"].to_numpy()
    y = np.log(np.maximum(trimmed["frequency"].to_numpy(), 1e-12))
    yhat = m * X + c
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def kfold_cv(df: pd.DataFrame, k: int, patience: int = 0, seed: int = 0) -> Dict[str, float]:
    """Return metrics from k-fold cross-validation with early stopping."""

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    fold_sizes = np.full(k, len(df) // k, dtype=int)
    fold_sizes[: len(df) % k] += 1
    start = 0
    val_history = []
    metrics = []
    for fold_idx, fold_size in enumerate(fold_sizes):
        stop = start + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        try:
            m, c, r2_train = fit_logfreq_vs_A(train_df)
            r2_val = eval_logfreq_vs_A(val_df, m, c)
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("Cross-validation fold %d skipped: %s", fold_idx, exc)
            start = stop
            continue
        metrics.append({"m": m, "c": c, "R2_train": r2_train, "R2_val": r2_val})
        val_history.append(r2_val)
        if patience and early_stopping(val_history, patience=patience):
            logger.info("Early stopping after %d folds", fold_idx + 1)
            break
        start = stop
    if not metrics:
        raise RuntimeError("No valid folds for cross-validation")
    best = max(metrics, key=lambda d: d["R2_val"])
    best["folds_evaluated"] = len(metrics)
    return best


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


def shuffle_frequencies(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a copy of ``df`` with frequencies randomly permuted.

    This serves as a simple control run that destroys any relationship between
    assembly size and frequency while preserving the marginal distribution of
    counts.
    """

    shuffled = df.copy()
    rng = np.random.default_rng(seed)
    shuffled["frequency"] = rng.permutation(shuffled["frequency"].to_numpy())
    return shuffled

if __name__ == "__main__":
    args = parser.parse_args()
    setup_reproducibility(args.seed)
    df = pd.read_csv(args.csv)
    df_base = shuffle_frequencies(df, seed=args.seed) if args.baseline else None
    if args.cv > 1:
        best = kfold_cv(df, args.cv, patience=args.patience, seed=args.seed)
        rho, p = degeneracy_spearman(df)
        result = {
            "m": best["m"],
            "R2_train": best["R2_train"],
            "R2_val": best["R2_val"],
            "rho_deg": rho,
            "p_deg": p,
            "folds": best["folds_evaluated"],
            "N_rows": int(df.shape[0]),
        }
        if args.baseline and df_base is not None:
            best_base = kfold_cv(df_base, args.cv, patience=args.patience, seed=args.seed)
            rho_b, p_b = degeneracy_spearman(df_base)
            result.update(
                {
                    "m_baseline": best_base["m"],
                    "R2_train_baseline": best_base["R2_train"],
                    "R2_val_baseline": best_base["R2_val"],
                    "rho_deg_baseline": rho_b,
                    "p_deg_baseline": p_b,
                }
            )
        logger.info("%s", result)
    else:
        m, c, r2 = fit_logfreq_vs_A(df)
        m_lo, m_hi = bootstrap_ci(df, alpha=args.alpha, B=args.boot)
        rho, p = degeneracy_spearman(df)
        result = {
            "m": m,
            "m_CI": [m_lo, m_hi],
            "R2": r2,
            "rho_deg": rho,
            "p_deg": p,
            "N_rows": int(df.shape[0]),
        }
        if args.baseline and df_base is not None:
            m_b, c_b, r2_b = fit_logfreq_vs_A(df_base)
            m_lo_b, m_hi_b = bootstrap_ci(df_base, alpha=args.alpha, B=args.boot)
            rho_b, p_b = degeneracy_spearman(df_base)
            result.update(
                {
                    "m_baseline": m_b,
                    "m_CI_baseline": [m_lo_b, m_hi_b],
                    "R2_baseline": r2_b,
                    "rho_deg_baseline": rho_b,
                    "p_deg_baseline": p_b,
                }
            )
        logger.info("%s", result)
