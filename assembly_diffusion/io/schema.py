"""CSV schema definitions and helpers.

The project logs protocol summaries as comma-separated values with a fixed
schema.  :func:`write_protocol_csv` ensures that rows are written in a stable
order and normalises numeric types for downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from typing import Iterable

import pandas as pd

# Column names and dtypes expected by downstream tooling.  The order here is
# important and is used when writing header rows.
PROTOCOL_COLUMNS = [
    "id",
    "universe",
    "grammar",
    "As_lower",
    "As_upper",
    "validity",
    "frequency",
    "d_min",
]

PROTOCOL_DTYPES: dict[str, str] = {
    "id": "string",
    "universe": "string",
    "grammar": "string",
    "As_lower": "int64",
    "As_upper": "int64",
    "validity": "float64",
    "frequency": "float64",
    "d_min": "float64",
}


@dataclass
class ProtocolRow:
    """Representation of a single protocol summary row."""

    id: str
    universe: str
    grammar: str
    As_lower: int
    As_upper: int
    validity: float
    frequency: float
    d_min: float


def write_protocol_csv(path: str, rows: Iterable[ProtocolRow]) -> None:
    """Write ``rows`` to ``path`` using the protocol CSV schema.

    Parameters
    ----------
    path:
        Destination CSV file path.
    rows:
        Iterable of :class:`ProtocolRow` instances that will be written in the
        given order.  Values are normalised to the dtypes specified in
        :data:`PROTOCOL_DTYPES`.
    """

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PROTOCOL_COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    str(r.id),
                    str(r.universe),
                    str(r.grammar),
                    int(r.As_lower),
                    int(r.As_upper),
                    float(r.validity),
                    float(r.frequency),
                    float(r.d_min),
                ]
            )


def read_protocol_csv(path: str) -> pd.DataFrame:
    """Load a protocol CSV file enforcing :data:`PROTOCOL_DTYPES`.

    Parameters
    ----------
    path:
        Location of the CSV file.
    """

    return pd.read_csv(path, dtype=PROTOCOL_DTYPES)
