"""Experiment specification for statistical analysis helpers.

baseline: expected results from simple arrays and SciPy/statsmodels serve as
    reference behavior.
data_sources: synthetic lists and pandas ``DataFrame`` objects defined within
    the tests.
method: invoke functions from :mod:`analysis` to compute statistical
    comparisons and diversity metrics.
metrics: Kolmogorov–Smirnov p-values, scaffold counts, regression
    parameters, bootstrap medians, sensitivity medians, error quantiles and
    calibration curves.
objective: verify the analysis utilities produce correct and stable
    statistics across a range of scenarios.
params: tests exercise options such as bootstrap iterations, quantiles and
    lambda values.
repro: deterministic NumPy operations and fixed random seeds yield
    reproducible outputs.
validation: running ``pytest tests/test_analysis.py`` executes the
    assertions below.
"""

import numpy as np
import pandas as pd
import pytest

from analysis import (
    ks_test,
    scaffold_diversity,
    mixed_effects_logistic,
    bootstrap_delta_median,
    sensitivity_over_lambda,
    error_quantiles,
    calibration_curve,
    train_val_test_split,
    early_stopping,
)


def test_ks_test():
    res = ks_test([1, 2, 3], [4, 5, 6])
    assert set(res.keys()) == {"statistic", "pvalue"}
    assert 0 <= res["pvalue"] <= 1


def test_scaffold_diversity():
    rdkit = pytest.importorskip("rdkit")
    smiles = ["c1ccccc1", "c1ccncc1"]  # benzene vs. pyridine
    assert scaffold_diversity(smiles) == 2


def test_mixed_effects_logistic():
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "x": [0.0, 0.0, 1.0, 1.0],
            "g": ["a", "a", "b", "b"],
        }
    )
    result = mixed_effects_logistic(df, "y ~ x", "g")
    assert "x" in result.params


def test_bootstrap_delta_median():
    a = [1, 2, 3]
    b = [2, 3, 4]
    diffs = bootstrap_delta_median(a, b, n_boot=200, random_state=0)
    expected = np.median(a) - np.median(b)
    assert np.abs(np.mean(diffs) - expected) < 0.5


def test_sensitivity_over_lambda():
    df = pd.DataFrame({"ai_exact": [1, 2, 3], "ai_surrogate": [3, 4, 5]})
    medians = sensitivity_over_lambda(df, lambdas=[0.25, 0.75, 1.0])
    assert medians[1.0] == pytest.approx(np.median(df["ai_exact"]))
    assert set(medians.keys()) == {0.25, 0.75, 1.0}


def test_error_quantiles():
    pred = [1.0, 2.0, 3.0]
    true = [1.0, 1.0, 4.0]
    qs = error_quantiles(pred, true, quantiles=[0.0, 0.5, 1.0])
    assert qs[0.0] == 0.0
    assert qs[0.5] == pytest.approx(1.0)
    assert qs[1.0] == pytest.approx(1.0)


def test_calibration_curve():
    pred = [0.0, 1.0, 2.0, 3.0]
    true = [0.0, 0.5, 2.0, 2.5]
    curve = calibration_curve(pred, true, bins=2)
    assert curve["pred_mean"].shape == curve["true_mean"].shape
    # First bin contains two lowest points with mean pred 0.5 and mean true 0.25
    assert curve["pred_mean"][0] == pytest.approx(0.5)
    assert curve["true_mean"][0] == pytest.approx(0.25)


def test_train_val_test_split():
    data = np.arange(20)
    train, val, test = train_val_test_split(
        data, train_size=0.6, val_size=0.2, test_size=0.2, random_state=0
    )
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    all_parts = np.concatenate((train, val, test))
    assert sorted(all_parts.tolist()) == list(data)


def test_early_stopping():
    history = [0.5, 0.6, 0.6, 0.59, 0.58]
    assert early_stopping(history, patience=2)
    assert not early_stopping(history[:3], patience=2)
