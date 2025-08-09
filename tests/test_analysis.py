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
