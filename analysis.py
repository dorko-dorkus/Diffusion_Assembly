"""Analysis utilities for the Diffusion Assembly project.

This module collects statistical routines used in the analysis notebook
specified in the project instructions.  The functions are lightweight so
that they can also be executed in continuous integration environments.
"""

from __future__ import annotations

from typing import Iterable, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

try:  # RDKit is optional and may not be installed in every environment.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised only when RDKit is missing
    Chem = None
    MurckoScaffold = None


def ks_test(sample_a: Sequence[float], sample_b: Sequence[float]) -> Dict[str, float]:
    """Return the Kolmogorov–Smirnov two-sample test statistic and p-value."""
    stat, pvalue = ks_2samp(sample_a, sample_b)
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def scaffold_diversity(smiles: Iterable[str]) -> int:
    """Count the number of unique Bemis–Murcko scaffolds in ``smiles``.

    Parameters
    ----------
    smiles:
        An iterable of SMILES strings.  RDKit must be installed in order to
        compute scaffolds.  Invalid molecules are skipped.
    """

    if Chem is None:
        raise ImportError("RDKit is required for scaffold diversity computation")

    scaffolds = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.add(Chem.MolToSmiles(scaffold))
    return len(scaffolds)


def mixed_effects_logistic(df: pd.DataFrame, formula: str, group_col: str):
    """Fit a mixed-effects logistic regression using :mod:`statsmodels`.

    The implementation uses Generalised Estimating Equations (GEE) with an
    exchangeable covariance structure which captures a random intercept for
    ``group_col``.  The returned object is a statsmodels results instance.
    """

    model = GEE.from_formula(formula, groups=group_col, data=df,
                             family=Binomial(), cov_struct=Exchangeable())
    result = model.fit()
    return result


def bootstrap_delta_median(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    n_boot: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """Bootstrap the difference in medians ``median(a) - median(b)``.

    Returns an array containing the bootstrap distribution of the delta
    median.  Users can compute confidence intervals from this array.
    """

    rng = np.random.default_rng(random_state)
    a = np.asarray(sample_a)
    b = np.asarray(sample_b)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_a = rng.choice(a, size=a.size, replace=True)
        boot_b = rng.choice(b, size=b.size, replace=True)
        diffs[i] = np.median(boot_a) - np.median(boot_b)
    return diffs


def sensitivity_over_lambda(
    df: pd.DataFrame, lambdas: Sequence[float] = (0.25, 0.75, 1.0)
) -> Dict[float, float]:
    """Compute median AI values for different mixing parameters ``lambda``.

    For each ``lambda`` the function forms ``lambda*ai_exact + (1-lambda)``
    ``* ai_surrogate`` and returns the median of the resulting series.
    ``df`` is expected to contain ``ai_exact`` and ``ai_surrogate`` columns.
    """

    medians = {}
    for lam in lambdas:
        mix = lam * df["ai_exact"] + (1 - lam) * df["ai_surrogate"]
        medians[lam] = float(mix.median())
    return medians


def calibration_curve(
    pred: Sequence[float],
    true: Sequence[float],
    bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute a simple calibration curve for predictions.

    The range of ``pred`` is split into ``bins`` equal-width segments.  For
    each non-empty bin the mean predicted value and the mean true value are
    recorded.  The function returns arrays of these means which can be used to
    plot calibration curves.
    """

    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same length")
    if p.size == 0 or bins <= 0:
        return {"pred_mean": np.array([]), "true_mean": np.array([])}

    edges = np.linspace(p.min(), p.max(), bins + 1)
    inds = np.digitize(p, edges) - 1
    pred_means = []
    true_means = []
    for b in range(bins):
        mask = inds == b
        if np.any(mask):
            pred_means.append(p[mask].mean())
            true_means.append(t[mask].mean())
    return {
        "pred_mean": np.asarray(pred_means, dtype=float),
        "true_mean": np.asarray(true_means, dtype=float),
    }


def error_quantiles(
    pred: Sequence[float],
    true: Sequence[float],
    quantiles: Sequence[float] | None = None,
) -> Dict[float, float]:
    """Return absolute error quantiles between ``pred`` and ``true`` values.

    Parameters
    ----------
    pred, true:
        Sequences of predictions and ground truth values of equal length.
    quantiles:
        Quantiles ``q`` in ``[0, 1]`` at which to evaluate the distribution of
        absolute errors ``|pred - true|``.  By default the function computes the
        deciles.
    """

    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 11)
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same length")
    if p.size == 0:
        return {float(q): 0.0 for q in quantiles}
    err = np.abs(p - t)
    qs = np.quantile(err, quantiles)
    return {float(q): float(v) for q, v in zip(quantiles, qs)}


__all__ = [
    "ks_test",
    "scaffold_diversity",
    "mixed_effects_logistic",
    "bootstrap_delta_median",
    "sensitivity_over_lambda",
    "calibration_curve",
    "error_quantiles",
]
