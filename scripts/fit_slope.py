#!/usr/bin/env python3
"""
Fit log-frequency vs A slope on aggregate data.

The script consumes ``agg.csv`` produced by ``aggregate.py`` and emits a
JSON summary with slope, intercept, R^2, bootstrap confidence interval, and
the number of bins used.

Flag tolerant: accepts ``--in``/``--input``/``-i`` and
``--out``/``--output``/``-o``; unrecognised arguments are ignored.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd


def _ap() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--input", dest="inp")
    ap.add_argument("-i", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--output", dest="out")
    ap.add_argument("-o", dest="out")
    ap.add_argument("--bootstrap", type=int, default=1000)
    return ap


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"A", "count", "frequency"}
    if not need.issubset(df.columns):
        sys.exit(f"agg.csv missing {need - set(df.columns)}")
    return df.sort_values("A").reset_index(drop=True)


def _trim90(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("A").copy()
    df["cum"] = df["frequency"].cumsum()
    lo = df[df["cum"] <= 0.05]["A"].max() if (df["cum"] <= 0.05).any() else df["A"].min()
    hi = df[df["cum"] >= 0.95]["A"].min() if (df["cum"] >= 0.95).any() else df["A"].max()
    return df[(df["A"] >= lo) & (df["A"] <= hi)]


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, m = beta[0], beta[1]
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return m, c, r2


def main() -> int:
    ap = _ap()
    args, unknown = ap.parse_known_args()
    if unknown:
        print(
            "WARN fit_slope.py: ignoring extra args:",
            " ".join(unknown),
            file=sys.stderr,
        )

    if not args.inp or not args.out:
        ap.print_help()
        sys.exit(2)

    df = _read(args.inp)
    df = df[df["frequency"] > 0].copy()
    if len(df) < 2:
        sys.exit("Not enough bins for slope.")
    df = _trim90(df)

    x = df["A"].to_numpy()
    y = np.log(df["frequency"].to_numpy())
    m, c, r2 = _ols(x, y)

    B = max(10, args.bootstrap)
    rng = np.random.default_rng(0)
    idx = np.arange(len(df))
    w = df["frequency"].to_numpy()
    w = w / w.sum()
    m_bs = []
    for _ in range(B):
        take = rng.choice(idx, size=len(idx), replace=True, p=w)
        mb, _, _ = _ols(x[take], y[take])
        m_bs.append(mb)
    lo, hi = np.percentile(m_bs, [2.5, 97.5]).tolist()

    out = {
        "m": float(m),
        "c": float(c),
        "r2": float(r2),
        "ci95": [float(lo), float(hi)],
        "used_bins": int(len(df)),
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

