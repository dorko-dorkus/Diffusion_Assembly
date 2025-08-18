import pandas as pd
import pytest

from loto.pricing.providers import CsvProvider, StaticCurveProvider, Em6Provider


def _sample_series():
    return pd.Series(
        [10.0, 20.0],
        index=pd.date_range(
            "2024-01-01", periods=2, freq="5min", tz="Pacific/Auckland"
        ),
        name="price",
    )


def _write_csv(path, series):
    df = series.reset_index()
    df.columns = ["time", "price"]
    # write times without timezone information
    df["time"] = df["time"].dt.tz_localize(None)
    df.to_csv(path, index=False)


def test_csv_provider_loads_series(tmp_path):
    csv_path = tmp_path / "prices.csv"
    _write_csv(csv_path, _sample_series())

    provider = CsvProvider(csv_path)
    result = provider.series()

    expected = _sample_series()
    pd.testing.assert_series_equal(result, expected)
    assert result.index.freq == pd.tseries.frequencies.to_offset("5min")
    assert str(result.index.tz) == "Pacific/Auckland"


def test_csv_provider_missing_file(tmp_path):
    provider = CsvProvider(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        provider.series()


def test_static_curve_provider():
    curve = _sample_series()
    provider = StaticCurveProvider(curve)
    pd.testing.assert_series_equal(provider.series(), curve)


def test_em6_provider_uses_cache(tmp_path):
    series = _sample_series()
    cache_file = tmp_path / "north.csv"
    _write_csv(cache_file, series)

    provider = Em6Provider(region="north", cache_dir=tmp_path)
    first = provider.series()
    cache_file.unlink()  # remove file to ensure cached result is used
    second = provider.series()

    pd.testing.assert_series_equal(first, second)


def test_em6_provider_bad_args(tmp_path):
    with pytest.raises(ValueError):
        Em6Provider()  # neither region nor node
    with pytest.raises(ValueError):
        Em6Provider(region="a", node="b")  # both provided

    provider = Em6Provider(region="missing", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        provider.series()
