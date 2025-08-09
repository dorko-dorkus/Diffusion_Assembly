import json
import os
from pathlib import Path

SCHEMA_VERSION = "1.0"


def write_metrics(outdir, *, valid_fraction, uniqueness, diversity, novelty, median_ai):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid_fraction": float(valid_fraction),
        "uniqueness": float(uniqueness),
        "diversity": float(diversity),
        "novelty": float(novelty),
        "median_ai": float(median_ai),
    }
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
