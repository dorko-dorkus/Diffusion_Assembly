import math
import numpy as np
from loto.scheduling.duration_models import Triangular, Lognormal, make_sampler

N = 50_000


def _check_stats(samples: np.ndarray, mean: float, std: float) -> None:
    assert abs(samples.mean() - mean) / mean < 0.02
    assert abs(samples.std(ddof=0) - std) / std < 0.05


def test_triangular_stats() -> None:
    rng = np.random.default_rng(0)
    dist = Triangular(1.0, 2.0, 4.0)
    samples = np.array([dist(rng) for _ in range(N)])
    mean = (1 + 2 + 4) / 3
    var = (1**2 + 2**2 + 4**2 - 1*2 - 1*4 - 2*4) / 18
    _check_stats(samples, mean, math.sqrt(var))
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    assert [dist(rng1) for _ in range(5)] == [dist(rng2) for _ in range(5)]


def test_lognormal_stats() -> None:
    rng = np.random.default_rng(0)
    dist = Lognormal(0.0, 0.5)
    samples = np.array([dist(rng) for _ in range(N)])
    mean = math.exp(0.5**2 / 2)
    std = math.sqrt((math.exp(0.5**2) - 1) * math.exp(0.5**2))
    _check_stats(samples, mean, std)
    rng1 = np.random.default_rng(321)
    rng2 = np.random.default_rng(321)
    assert [dist(rng1) for _ in range(5)] == [dist(rng2) for _ in range(5)]


def test_factory_scaling() -> None:
    context = {"health": 0.5, "access": 1.0, "experience": 1.0}
    sampler = make_sampler("triangular", context)
    rng = np.random.default_rng(0)
    samples = np.array([sampler(rng) for _ in range(N)])
    base_mean = (1 + 2 + 4) / 3
    base_std = math.sqrt((1**2 + 2**2 + 4**2 - 1*2 - 1*4 - 2*4) / 18)
    scale = 1 / (0.5 * 1.0 * 1.0)
    _check_stats(samples, base_mean * scale, base_std * scale)


def test_factory_scaling_lognormal() -> None:
    context = {"health": 1.0, "access": 0.5, "experience": 0.5}
    sampler = make_sampler("lognormal", context)
    rng = np.random.default_rng(0)
    samples = np.array([sampler(rng) for _ in range(N)])
    base_mean = math.exp(0.5**2 / 2)
    base_std = math.sqrt((math.exp(0.5**2) - 1) * math.exp(0.5**2))
    scale = 1 / (1.0 * 0.5 * 0.5)
    _check_stats(samples, base_mean * scale, base_std * scale)
