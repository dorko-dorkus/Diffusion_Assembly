import json, os, subprocess, sys
from pathlib import Path

def test_pipeline_runs():
    # Use ab_compare script which calls experiment twice and writes metrics.json
    out = subprocess.check_output([sys.executable, "scripts/ab_compare.py"]).decode()
    assert "ab_summary.json" in out or os.path.exists("results/ab_summary.json")

    # Check the last two runs contain metrics.json with required keys
    results = sorted(Path("results").glob("exp*"), key=os.path.getmtime)
    assert len(results) >= 2
    for run_dir in results[-2:]:
        m = json.load(open(run_dir / "metrics.json"))
        for k in [
            "valid_fraction",
            "uniqueness",
            "diversity",
            "novelty",
            "qed_mean",
            "qed_median",
            "sa_mean",
            "sa_median",
            "A_hat_median",
            "A_hat_IQR",
            "A_hat_p10",
            "A_hat_p90",
            "seed",
            "config",
            "schema_version",
        ]:
            assert k in m
        # calibration curve should be present
        cal_path = run_dir / "ai_calibration.csv"
        assert cal_path.exists()
        assert cal_path.read_text().startswith("quantile")
