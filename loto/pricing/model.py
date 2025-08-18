from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Any

import numpy as np


@dataclass(frozen=True)
class PriceSeries:
    """Immutable price curve defined on integer buckets."""

    buckets: tuple[float, ...]
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.buckets) != len(self.prices):
            raise ValueError("buckets and prices must have the same length")

    def interp(self, new_buckets: Sequence[float]) -> "PriceSeries":
        """Linearly interpolate to ``new_buckets``."""

        x = np.asarray(self.buckets, dtype=float)
        y = np.asarray(self.prices, dtype=float)
        nx = np.asarray(list(new_buckets), dtype=float)
        ny = np.interp(nx, x, y)
        return PriceSeries(tuple(nx.tolist()), tuple(ny.tolist()))


def _normalise(value: Any, buckets: Sequence[float]) -> PriceSeries:
    """Normalise ``value`` to a :class:`PriceSeries` on ``buckets``.

    ``value`` may be a scalar, a sequence matching ``buckets``, a mapping
    of bucket to price, or an existing :class:`PriceSeries` which will be
    interpolated onto ``buckets``.
    """

    b = tuple(float(b) for b in buckets)

    if isinstance(value, PriceSeries):
        return value.interp(b)

    if np.isscalar(value):  # type: ignore[arg-type]
        return PriceSeries(b, tuple(float(value) for _ in b))

    if isinstance(value, Mapping):
        keys = sorted(float(k) for k in value.keys())
        prices = [float(value[k]) for k in keys]
        return PriceSeries(tuple(keys), tuple(prices)).interp(b)

    if isinstance(value, Sequence):
        if len(value) != len(buckets):
            raise ValueError("Length of sequence must match buckets")
        return PriceSeries(b, tuple(float(v) for v in value))

    raise TypeError("Unsupported price input")


@dataclass(frozen=True)
class PriceModel:
    """Container for low/medium/high price scenarios."""

    low: PriceSeries
    med: PriceSeries
    high: PriceSeries

    @classmethod
    def build(
        cls,
        buckets: Sequence[float],
        low: Any,
        med: Any,
        high: Any,
    ) -> "PriceModel":
        b = tuple(float(b) for b in buckets)
        return cls(
            low=_normalise(low, b),
            med=_normalise(med, b),
            high=_normalise(high, b),
        )

    def sample(
        self,
        scenario: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> PriceSeries:
        """Sample one of the scenarios.

        If ``scenario`` is provided, it must be one of ``"low"``,
        ``"med"`` or ``"high"``. Otherwise a scenario is chosen uniformly
        at random.
        """

        if scenario is None:
            rng = rng or np.random.default_rng()
            scenario = rng.choice(["low", "med", "high"])  # type: ignore[assignment]
        if scenario not in {"low", "med", "high"}:
            raise ValueError("scenario must be 'low', 'med' or 'high'")
        return getattr(self, scenario)


__all__ = ["PriceSeries", "PriceModel"]
