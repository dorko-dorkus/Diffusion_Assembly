import subprocess
import sys
import pandas as pd

from assembly_diffusion.calibrators import summarize_dmin


def _sample_df():
    data = {
        "id": list(range(10)),
        "universe": ["S"] * 10,
        "grammar": ["G"] * 10,
        "As_lower": [0] * 10,
        "As_upper": [1] * 10,
        "validity": [1] * 10,
        "d_min": list(range(1, 10)) + [None],
        "frequency": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    }
    return pd.DataFrame(data)


def test_summarize_dmin(tmp_path):
    df = _sample_df()
    summary, frac = summarize_dmin(df)
    assert frac == 0.9
    assert summary.shape[0] == 1
    assert summary.loc[0, "rho"] > 0
    assert summary.loc[0, "p"] < 0.05

    csv_path = tmp_path / "calibs.csv"
    out_path = tmp_path / "summary.csv"
    df.to_csv(csv_path, index=False)
    subprocess.check_call([sys.executable, "scripts/summarize_dmin.py", str(csv_path), "--out", str(out_path)])
    out_df = pd.read_csv(out_path)
    assert "rho" in out_df.columns
