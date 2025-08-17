import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "demo"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name)


def test_schema_and_uniqueness():
    specs = {
        "assets.csv": ["tag", "type", "description"],
        "line_list.csv": [
            "tag",
            "from_tag",
            "to_tag",
            "service",
            "direction",
            "description",
        ],
        "valves.csv": ["tag", "line_tag", "type", "bypass_group"],
        "drains_vents.csv": ["tag", "line_tag", "dv_type"],
        "energy_sources.csv": ["asset_tag", "line_tag"],
        "air_map.csv": ["valve_tag", "air_line_tag", "air_source_tag"],
    }
    for fname, cols in specs.items():
        df = _read(fname)
        assert list(df.columns) == cols
        if "tag" in df.columns:
            assert df["tag"].is_unique
    lines = _read("line_list.csv")
    assert lines["direction"].notna().all()
    assert (lines["direction"].str.len() > 0).all()


def test_cross_file_references():
    assets = _read("assets.csv")
    lines = _read("line_list.csv")
    valves = _read("valves.csv")
    drains = _read("drains_vents.csv")
    energy = _read("energy_sources.csv")
    air = _read("air_map.csv")

    asset_tags = set(assets["tag"])
    line_tags = set(lines["tag"])
    valve_tags = set(valves["tag"])

    assert set(lines["from_tag"]).issubset(asset_tags)
    assert set(lines["to_tag"]).issubset(asset_tags)

    assert set(valves["line_tag"]).issubset(line_tags)
    assert set(drains["line_tag"]).issubset(line_tags)

    assert set(energy["asset_tag"]).issubset(asset_tags)
    assert set(energy["line_tag"]).issubset(line_tags)

    assert set(air["valve_tag"]).issubset(valve_tags)
    assert set(air["air_line_tag"]).issubset(line_tags)
    assert set(air["air_source_tag"]).issubset(asset_tags)

    for tag, service in lines[["tag", "service"]].itertuples(index=False):
        if service in {"Steam", "InstrumentAir"}:
            assert tag in set(energy["line_tag"])


def test_mini_plant_content():
    assets = _read("assets.csv")
    lines = _read("line_list.csv")
    valves = _read("valves.csv")

    assert {"Steam", "Condensate", "InstrumentAir"}.issubset(set(lines["service"]))

    groups = valves["bypass_group"].value_counts()
    assert (groups.get("BG-1", 0)) >= 2

    pt_rows = assets[assets["type"] == "PressureTransmitter"]
    assert len(pt_rows) == 1
    pt_tag = pt_rows.iloc[0]["tag"]
    target_tag = "HX-100"
    connected = (
        ((lines["from_tag"] == pt_tag) & (lines["to_tag"] == target_tag))
        | ((lines["to_tag"] == pt_tag) & (lines["from_tag"] == target_tag))
    )
    assert lines[connected & (lines["service"] == "Steam")].shape[0] == 1
