from __future__ import annotations

from .model import PriceSeries


def hedge_price(
    spot: PriceSeries,
    hedge: PriceSeries,
    exposure: float,
) -> PriceSeries:
    """Blend ``spot`` with ``hedge`` using exposure ``alpha``.

    The returned series is defined on the spot's buckets and is computed
    as ``alpha * spot + (1 - alpha) * hedge`` where ``alpha`` is the
    provided ``exposure``.
    """

    if not 0.0 <= exposure <= 1.0:
        raise ValueError("exposure must be within [0, 1]")

    hedge_aligned = hedge.interp(spot.buckets)
    prices = tuple(
        exposure * s + (1.0 - exposure) * h
        for s, h in zip(spot.prices, hedge_aligned.prices)
    )
    return PriceSeries(spot.buckets, prices)


__all__ = ["hedge_price"]
