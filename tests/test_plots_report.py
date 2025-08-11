import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_plots_and_report(tmp_path, monkeypatch):
    df = pd.DataFrame({"A": [1, 2, 3], "count": [10, 5, 1], "frequency": [0.625, 0.3125, 0.0625]})
    agg = tmp_path / "agg.csv"
    df.to_csv(agg, index=False)
    outdir = tmp_path / "out"

    monkeypatch.setenv("TRIALS", "123")
    monkeypatch.setenv("ASSEMBLYMC_BIN", "AssemblyMC.exe")
    monkeypatch.setenv("ASSEMBLYMC_COMMIT", "abc123")

    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "plots.py"),
                    "--in", str(agg), "--outdir", str(outdir)], check=True)

    exp_imgs = ["freq_log_vs_A.png", "survival_SA.png", "freq_vs_A.png", "cdf_A.png"]
    for name in exp_imgs:
        assert (outdir / name).exists()
    assert (outdir / "report.html").exists()
    assert (outdir / "report.pdf").exists()
    data = json.loads((outdir / "report.json").read_text())
    assert data["assemblymc"]["commit"] == "abc123"
    assert "--trials 123" in data["assemblymc"]["cmd"]
    assert data["assemblymc"]["bin"] == "AssemblyMC.exe"
