"""Tests for baseline_slope_fit utility."""
import numpy as np

from analysis import baseline_slope_fit


def test_baseline_slope_fit(tmp_path):
    rng = np.random.default_rng(0)
    # Generate 6000 samples from a Poisson distribution with mean 10
    samples = rng.poisson(lam=10, size=6000)
    out_path = tmp_path / "boots.npy"
    res = baseline_slope_fit(samples, n_boot=200, save_bootstrap=str(out_path), random_state=0)
    assert res["m"] < 0
    assert res["ci_hi"] < 0
    data = np.load(out_path)
    assert data.shape[0] == 200
