import numpy as np
from loto.scheduling.expectation import estimate


def test_ci_shrinks_and_targets() -> None:
    def sampler(rng: np.random.Generator):
        j = rng.normal()
        return j, {"gt0": j > 0}

    res_small = estimate(sampler, rng=np.random.default_rng(0), max_samples=10, ci_threshold=0)
    res_large = estimate(sampler, rng=np.random.default_rng(0), max_samples=200, ci_threshold=0)

    width_small = res_small.ci[1] - res_small.ci[0]
    width_large = res_large.ci[1] - res_large.ci[0]

    assert width_large < width_small
    assert abs(res_large.p_targets["gt0"] - 0.5) < 0.1


def test_early_stopping() -> None:
    def sampler(_: np.random.Generator):
        return 1.0, {"gt0": True}

    res = estimate(
        sampler,
        rng=np.random.default_rng(0),
        max_samples=1000,
        ci_threshold=0.01,
    )

    assert res.stopped_early
    assert res.n < 1000
    assert res.ci == (1.0, 1.0)
    assert res.p_targets["gt0"] == 1.0
