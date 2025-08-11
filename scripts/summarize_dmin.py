from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root on path when executed as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from assembly_diffusion.calibrators.dmin_summary import summarize_dmin


parser = argparse.ArgumentParser(description="Summarize d_min estimates from calibrator outputs")
parser.add_argument("csv", help="Input calibrator CSV with frequency and d_min columns")
parser.add_argument("--out", default="dmin_summary.csv", help="Output summary CSV path")


if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    summary, frac = summarize_dmin(df)
    summary.to_csv(args.out, index=False)
    if frac < 0.9:
        raise SystemExit(f"d_min non-null fraction {frac:.3f} < 0.9")
    if summary.empty or (summary["rho"] <= 0).any() or (summary["p"] >= 0.05).any():
        raise SystemExit("Spearman correlation did not meet criteria")
