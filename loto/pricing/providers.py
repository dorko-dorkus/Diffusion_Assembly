from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class CsvProvider:
    """Load pricing data from a CSV file.

    The CSV file must contain ``time`` and ``price`` columns. ``time``
    values are parsed as datetimes and normalised to 5‑minute buckets in
    the ``Pacific/Auckland`` timezone.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._series: Optional[pd.Series] = None

    def series(self) -> pd.Series:
        if self._series is not None:
            return self._series
        if not self.path.exists():
            raise FileNotFoundError(f"No such file: {self.path}")
        df = pd.read_csv(self.path)
        if "time" not in df.columns or "price" not in df.columns:
            raise ValueError("CSV must contain 'time' and 'price' columns")
        idx = pd.to_datetime(df["time"])
        idx.name = None
        tz = "Pacific/Auckland"
        if idx.dt.tz is None:
            idx = idx.dt.tz_localize(tz)
        else:
            idx = idx.dt.tz_convert(tz)
        s = pd.Series(df["price"].values, index=idx, name="price").sort_index()
        s = s.resample("5min").ffill()
        self._series = s
        return s


class StaticCurveProvider:
    """Return a pre-specified price curve."""

    def __init__(self, curve: pd.Series):
        if not isinstance(curve, pd.Series):
            raise TypeError("curve must be a pandas Series")
        self._curve = curve

    def series(self) -> pd.Series:
        return self._curve


class Em6Provider:
    """Stubbed Electricity Market (EM6) pricing provider.

    Real EM6 access would require network calls. For testing purposes the
    provider simply loads pricing data from a CSV cache. The cache
    filename is derived from the supplied region or node.
    """

    def __init__(self, *, region: Optional[str] = None, node: Optional[str] = None, cache_dir: Path | str = "."):
        if (region is None) == (node is None):
            raise ValueError("Exactly one of region or node must be provided")
        self.identifier = region or node
        self.cache_dir = Path(cache_dir)
        self._series: Optional[pd.Series] = None

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self.identifier}.csv"

    def series(self) -> pd.Series:
        if self._series is not None:
            return self._series
        path = self._cache_path()
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found for '{self.identifier}': {path}")
        self._series = CsvProvider(path).series()
        return self._series
