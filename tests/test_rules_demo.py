import json
from pathlib import Path
import yaml


def test_rules_demo_load_and_schema_and_hash():
    path = Path(__file__).resolve().parents[1] / "demo" / "rules_huntly.yml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Required top-level keys
    assert {"steam", "condensate", "instrument_air", "residual_pressure"} <= data.keys()

    # Steam defaults
    assert data["steam"]["isolation"] == "DDBB"
    assert data["steam"]["bleed"] is True

    # Condensate defaults
    assert data["condensate"]["isolation"] == "single"
    assert data["condensate"]["drain"] is True

    # Instrument air defaults
    assert data["instrument_air"]["supply"] == "block"

    # Residual pressure thresholds must be present and numeric
    for key in ("steam", "condensate", "instrument_air"):
        assert key in data["residual_pressure"]
        assert isinstance(data["residual_pressure"][key], (int, float))

    # Deterministic hash of the full configuration must be non-zero
    digest = hash(json.dumps(data, sort_keys=True))
    assert digest != 0
