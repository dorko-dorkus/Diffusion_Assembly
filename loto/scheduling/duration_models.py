from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
import math
import numpy as np


@dataclass(frozen=True)
class Triangular:
    """Triangular distribution with parameters ``a`` ≤ ``m`` ≤ ``b``."""

    a: float
    m: float
    b: float

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not (self.a <= self.m <= self.b):
            raise ValueError("Require a ≤ m ≤ b")

    def scale(self, factor: float) -> "Triangular":
        """Return a new distribution with scaled parameters."""
        return Triangular(self.a * factor, self.m * factor, self.b * factor)

    def __call__(self, rng: np.random.Generator) -> float:
        """Sample a value in *seconds* using ``rng``."""
        return float(rng.triangular(self.a, self.m, self.b))


@dataclass(frozen=True)
class Lognormal:
    """Log-normal distribution parameterised by ``mu`` and ``sigma``."""

    mu: float
    sigma: float

    def scale(self, factor: float) -> "Lognormal":
        """Return a new distribution scaled by ``factor``."""
        return Lognormal(self.mu + math.log(factor), self.sigma)

    def __call__(self, rng: np.random.Generator) -> float:
        """Sample a value in *seconds* using ``rng``."""
        return float(rng.lognormal(self.mu, self.sigma))


# Base sampler definitions used by :func:`make_sampler`.
_BASE_SAMPLERS: Mapping[str, Callable[[], object]] = {
    "triangular": lambda: Triangular(1.0, 2.0, 4.0),
    "lognormal": lambda: Lognormal(0.0, 0.5),
}


def make_sampler(class_id: str, context: Mapping[str, float]) -> Callable[[np.random.Generator], float]:
    """Factory returning a sampler adjusted for ``context``.

    Parameters
    ----------
    class_id:
        Identifier of the base sampler. Currently ``"triangular"`` and
        ``"lognormal"`` are supported.
    context:
        Mapping providing ``health``, ``access`` and ``experience`` scores. Each
        value should be positive; lower scores imply longer durations.
    """

    if class_id not in _BASE_SAMPLERS:
        raise KeyError(f"Unknown sampler class_id '{class_id}'")

    h = context.get("health", 1.0)
    a = context.get("access", 1.0)
    e = context.get("experience", 1.0)
    if h <= 0 or a <= 0 or e <= 0:  # pragma: no cover - basic validation
        raise ValueError("Context scores must be positive")

    scale = 1.0 / (h * a * e)

    base = _BASE_SAMPLERS[class_id]()
    scaled = base.scale(scale)
    return scaled


__all__ = ["Triangular", "Lognormal", "make_sampler"]
