from __future__ import annotations

import numpy as np
import pytest

from loto.pricing.model import PriceModel, PriceSeries
from loto.pricing.hedge import hedge_price


def test_interpolation_to_schedule_buckets() -> None:
    buckets = [0, 1, 2]
    model = PriceModel.build(
        buckets,
        low=10.0,
        med={0: 9.0, 2: 11.0},
        high=[8.0, 9.0, 10.0],
    )
    series = model.sample("med")
    assert series.buckets == (0.0, 1.0, 2.0)
    assert series.prices == pytest.approx((9.0, 10.0, 11.0))


def test_sampler_returns_known_scenario() -> None:
    model = PriceModel.build([0, 1], low=1, med=2, high=3)
    rng = np.random.default_rng(0)
    s = model.sample(rng=rng)
    assert s in [model.low, model.med, model.high]


def test_hedged_price_blend() -> None:
    spot = PriceSeries((0.0, 1.0, 2.0), (10.0, 20.0, 30.0))
    hedge = PriceSeries((0.0, 2.0), (8.0, 14.0))
    blended = hedge_price(spot, hedge, exposure=0.25)
    expected = (
        0.25 * 10.0 + 0.75 * 8.0,
        0.25 * 20.0 + 0.75 * 11.0,
        0.25 * 30.0 + 0.75 * 14.0,
    )
    assert blended.buckets == spot.buckets
    assert blended.prices == pytest.approx(expected)
