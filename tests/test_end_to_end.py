import json, os, subprocess, sys
from pathlib import Path

def test_smoke_ab(tmp_path):
    # run a tiny A/B with small n to stay fast
    out = subprocess.check_output([sys.executable, "scripts/ab_compare.py"]).decode()
    assert "ab_summary.json" in out or os.path.exists("results/ab_summary.json")
    s = json.load(open("results/ab_summary.json"))
    # loose sanity constraints for smoke test
    assert abs(s["validity_delta"]) <= 0.05
