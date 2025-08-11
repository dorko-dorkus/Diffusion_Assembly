#!/usr/bin/env python3
"""
Aggregate per-molecule AI results into A-bin counts for slope fitting.

Input (from compute_ai.py):
    id,smiles,As_lower,As_upper,d_min,validity,method,status,elapsed_ms
Output (--out): CSV with columns:
    A,count,frequency,valid_n,invalid_n

The interface is flag tolerant: ``--in``/``--input``/``-i`` and
``--out``/``--output``/``-o`` are accepted, while unknown additional
arguments are ignored for forward compatibility.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.run_logger import init_run_logger


def _arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser accepting several flag synonyms."""

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--input", dest="inp")
    ap.add_argument("-i", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--output", dest="out")
    ap.add_argument("-o", dest="out")
    ap.add_argument("--universe", default="M")
    ap.add_argument("--grammar", default=None)
    ap.add_argument("--log", dest="log", default=None)
    ap.add_argument("--grammar-text", dest="grammar_text", default=None)
    return ap


def _read_any(p: str) -> pd.DataFrame:
    """Read CSV/TSV/Parquet depending on extension."""

    path = pathlib.Path(p)
    if not path.exists():
        sys.exit(f"--in not found: {path}")
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    sys.exit(f"Unsupported input extension: {ext}")


def _mid_A(lo: int, hi: int) -> int:
    """Return midpoint rounded to nearest integer."""

    return int(round((int(lo) + int(hi)) / 2.0))


def main() -> int:
    ap = _arg_parser()
    args, unknown = ap.parse_known_args()
    if unknown:
        print(
            "WARN aggregate.py: ignoring extra args:",
            " ".join(unknown),
            file=sys.stderr,
        )

    if not args.inp or not args.out:
        ap.print_help()
        sys.exit(2)

    df = _read_any(args.inp)
    required = {"id", "smiles", "As_lower", "As_upper", "validity", "status", "method"}
    if not required.issubset(df.columns):
        sys.exit(f"Input missing columns: {sorted(required - set(df.columns))}")

    df = df.copy()
    df["A"] = [_mid_A(lo, hi) for lo, hi in zip(df["As_lower"], df["As_upper"])]
    df["valid"] = (df["validity"].astype(int) == 1) & (df["status"] == "ok")
    df["invalid"] = ~df["valid"]

    g = df.groupby("A", as_index=False).agg(
        count=("A", "size"),
        valid_n=("valid", "sum"),
        invalid_n=("invalid", "sum"),
    )
    total = float(len(df)) if len(df) else 1.0
    g["frequency"] = g["count"] / total
    g = g.sort_values("A").reset_index(drop=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False)

    # Produce protocol-ready per-object CSV for downstream integration.
    try:
        proto = df.merge(
            df.groupby("smiles")["smiles"].transform("count").rename("smiles_count"),
            left_index=True,
            right_index=True,
            how="left",
        )
    except Exception:
        proto = df.copy()
        proto["smiles_count"] = 1
    proto["frequency"] = proto["smiles_count"] / total
    grammar = args.grammar or ("G_MC" if (df["method"] == "assemblymc").any() else "G")

    run_log = None
    if args.log:
        run_log = init_run_logger(
            args.log,
            grammar=grammar,
            config={"input": args.inp, "output": args.out, "universe": args.universe},
            grammar_text=args.grammar_text,
        )
    proto_out = out.with_name("protocol_objects.csv")
    cols = ["id", "smiles", "As_lower", "As_upper", "validity", "frequency", "d_min"]
    for c in cols:
        if c not in proto.columns:
            proto[c] = np.nan
    proto = proto[cols]
    proto.insert(1, "universe", args.universe)
    proto.insert(2, "grammar", grammar)
    proto.to_csv(proto_out, index=False)

    msg = (
        f"bins={len(g)} total={int(total)} "
        f"wrote_agg={out} wrote_protocol={proto_out}"
    )
    if run_log:
        run_log.info(msg)
    print(msg)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
