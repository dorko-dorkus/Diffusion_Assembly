from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import math
import numpy as np


@dataclass
class ExpectationResult:
    """Result of a Monte Carlo expectation estimate."""

    mean: float
    ci: Tuple[float, float]
    n: int
    stderr: float
    stopped_early: bool
    p_targets: Dict[str, float]


Z_95 = 1.96


def estimate(
    sampler: Callable[[np.random.Generator], Tuple[float, Dict[str, float]]],
    *,
    rng: np.random.Generator | None = None,
    max_samples: int = 1000,
    ci_threshold: float | None = None,
) -> ExpectationResult:
    """Estimate ``E[J]`` via Monte Carlo sampling.

    Parameters
    ----------
    sampler:
        Callable that given a ``numpy.random.Generator`` returns a tuple of
        ``(J, p_targets)`` where ``J`` is the sample value and ``p_targets`` is a
        mapping of event names to indicator values.
    rng:
        Optional random number generator.  If ``None`` a default RNG is created.
    max_samples:
        Maximum number of samples to draw.
    ci_threshold:
        Optional half-width threshold for the 95% confidence interval.  If set,
        sampling stops early once the CI half-width is below this value.

    Returns
    -------
    ExpectationResult
        The estimated statistics.
    """

    rng = np.random.default_rng() if rng is None else rng

    samples: list[float] = []
    counts: Dict[str, float] = {}
    stopped_early = False

    for i in range(1, max_samples + 1):
        j, targets = sampler(rng)
        samples.append(j)
        for name, val in targets.items():
            counts[name] = counts.get(name, 0.0) + float(val)

        if ci_threshold is not None and i >= 2:
            stderr = float(np.std(samples, ddof=1) / math.sqrt(i))
            half_ci = Z_95 * stderr
            if half_ci < ci_threshold:
                stopped_early = True
                break

    n = len(samples)
    mean = float(np.mean(samples))
    stderr = float(np.std(samples, ddof=1) / math.sqrt(n)) if n >= 2 else float("inf")
    half_ci = Z_95 * stderr if n >= 2 else float("inf")
    ci = (mean - half_ci, mean + half_ci)

    p_targets = {name: val / n for name, val in counts.items()} if n else {}

    return ExpectationResult(mean, ci, n, stderr, stopped_early, p_targets)


__all__ = ["estimate", "ExpectationResult"]
