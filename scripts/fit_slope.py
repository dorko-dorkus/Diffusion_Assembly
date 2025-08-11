"""Fit slope analysis for unguided baseline samples.

This command fits a log-frequency vs. assembly index slope for a column ``A``
contained in a CSV file.  A bootstrap distribution of slope estimates is saved
to ``--out``.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from analysis import baseline_slope_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit baseline slope")
    parser.add_argument("csv", help="CSV file with column of assembly indices")
    parser.add_argument(
        "--column",
        default="A",
        help="Name of column containing assembly indices",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--boot", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument(
        "--out",
        type=str,
        default="bootstrap.npy",
        help="Path to save bootstrap slope estimates",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    if args.column not in df:
        raise KeyError(f"Column {args.column} not found in {args.csv}")
    values = df[args.column].to_numpy()
    result = baseline_slope_fit(
        values,
        alpha=args.alpha,
        n_boot=args.boot,
        save_bootstrap=args.out,
    )
    logger.info(
        "m=%f 95%% CI=[%f, %f] saved=%s",
        result["m"],
        result["ci_lo"],
        result["ci_hi"],
        args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
