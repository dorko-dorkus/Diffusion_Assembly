from dataclasses import asdict

import pandas as pd
from pandas.api.types import (
    is_string_dtype,
    is_integer_dtype,
    is_float_dtype,
)

from assembly_diffusion.io import (
    ProtocolRow,
    PROTOCOL_COLUMNS,
    PROTOCOL_DTYPES,
    write_protocol_csv,
    read_protocol_csv,
)


def test_protocol_csv_roundtrip(tmp_path):
    rows = [
        ProtocolRow(
            id="mol1",
            universe="S",
            grammar="g1",
            As_lower=1,
            As_upper=2,
            validity=0.5,
            frequency=3.0,
            d_min=0.1,
        ),
        ProtocolRow(
            id="mol2",
            universe="T",
            grammar="g2",
            As_lower=3,
            As_upper=3,
            validity=1.0,
            frequency=1.5,
            d_min=0.2,
        ),
    ]
    path = tmp_path / "protocol.csv"
    write_protocol_csv(path, rows)

    # Column order check
    df_plain = pd.read_csv(path)
    assert list(df_plain.columns) == PROTOCOL_COLUMNS

    # Type check with enforced schema
    df = read_protocol_csv(path)
    expected = pd.DataFrame([asdict(r) for r in rows]).astype(PROTOCOL_DTYPES)
    pd.testing.assert_frame_equal(df, expected)

    assert is_string_dtype(df["id"])
    assert is_string_dtype(df["universe"])
    assert is_string_dtype(df["grammar"])
    assert is_integer_dtype(df["As_lower"])
    assert is_integer_dtype(df["As_upper"])
    assert is_float_dtype(df["validity"])
    assert is_float_dtype(df["frequency"])
    assert is_float_dtype(df["d_min"])
