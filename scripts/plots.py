#!/usr/bin/env python3
"""
Make baseline plots for the unguided run.

Inputs:
  --in / --input / -i          Path to agg.csv with columns: A,count,frequency
  --outdir / --output / -o     Directory to write figures
  --slope PATH                 Optional path to slope.json (with keys m,c,ci95)
  --format {png,svg}           Default: png
  --dpi INT                    Default: 120
(Unknown extra args are ignored.)

Outputs (in --outdir):
  freq_log_vs_A.<ext>   - ln(frequency) vs A (with OLS line if slope.json present)
  survival_SA.<ext>     - Survival S(A) = 1 - CDF(A)
  freq_vs_A.<ext>       - frequency vs A (linear scale)
  cdf_A.<ext>           - cumulative distribution F(A)
  report.{html,pdf,json} - lightweight HTML/PDF report with provenance JSON
"""
from __future__ import annotations
import argparse, json, sys, pathlib, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _ap():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--input", dest="inp")
    ap.add_argument("-i", dest="inp")
    ap.add_argument("--outdir", dest="outdir")
    ap.add_argument("--output", dest="outdir")
    ap.add_argument("-o", dest="outdir")
    ap.add_argument("--slope", dest="slope")
    ap.add_argument("--format", dest="fmt", default="png", choices=["png", "svg"])
    ap.add_argument("--dpi", type=int, default=120)
    return ap


def _read_agg(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    need = {"A", "count", "frequency"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"agg.csv missing columns: {sorted(missing)}")
    df = df.sort_values("A").reset_index(drop=True)
    df["frequency"] = df["frequency"].clip(lower=1e-12)
    return df


def _maybe_read_slope(p: pathlib.Path | None):
    if not p:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _default_slope_path(agg_path: pathlib.Path) -> pathlib.Path:
    return agg_path.with_name("slope.json")


def _ln(x):
    return np.log(np.asarray(x, dtype=float))


def plot_freq_ln_vs_A(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    fmt: str,
    dpi: int,
    slope: dict | None,
    pdf: PdfPages | None = None,
):
    A = df["A"].to_numpy()
    f = df["frequency"].to_numpy()
    y = _ln(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(A, y, s=14)
    ax.set_xlabel("Assembly index A")
    ax.set_ylabel("ln frequency")
    ax.set_title("ln(frequency) vs A (unguided)")

    if slope and all(k in slope for k in ("m", "c")):
        m = float(slope["m"])
        c = float(slope["c"])
        xs = np.linspace(A.min(), A.max(), 100)
        ys = c + m * xs
        ax.plot(xs, ys, linewidth=1.5)
        ci = slope.get("ci95", None)
        if ci and len(ci) == 2:
            ax.text(0.01, 0.02,
                    f"m={m:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
                    transform=ax.transAxes)
        else:
            ax.text(0.01, 0.02, f"m={m:.3f}", transform=ax.transAxes)

    out = outdir / f"freq_log_vs_A.{fmt}"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return out


def plot_survival(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    fmt: str,
    dpi: int,
    pdf: PdfPages | None = None,
):
    d = df.sort_values("A").copy()
    d["cum"] = d["frequency"].cumsum()
    d["S"] = 1.0 - d["cum"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(d["A"].to_numpy(), d["S"].to_numpy(), where="post")
    ax.set_xlabel("Assembly index A")
    ax.set_ylabel("Survival S(A) = 1 − CDF(A)")
    ax.set_title("Survival function S(A)")
    out = outdir / f"survival_SA.{fmt}"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return out


def plot_freq_vs_A(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    fmt: str,
    dpi: int,
    pdf: PdfPages | None = None,
):
    """Plot frequency vs A on a linear scale."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df["A"].to_numpy(), df["frequency"].to_numpy(), width=0.8)
    ax.set_xlabel("Assembly index A")
    ax.set_ylabel("frequency")
    ax.set_title("frequency vs A")
    out = outdir / f"freq_vs_A.{fmt}"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return out


def plot_cdf(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    fmt: str,
    dpi: int,
    pdf: PdfPages | None = None,
):
    """Plot cumulative distribution F(A)."""

    d = df.sort_values("A").copy()
    d["F"] = d["frequency"].cumsum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(d["A"].to_numpy(), d["F"].to_numpy(), where="post")
    ax.set_xlabel("Assembly index A")
    ax.set_ylabel("CDF F(A)")
    ax.set_title("Cumulative distribution F(A)")
    out = outdir / f"cdf_A.{fmt}"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return out


def main():
    ap = _ap()
    args, unknown = ap.parse_known_args()
    if unknown:
        print("WARN plots.py: ignoring extra args:", " ".join(unknown), file=sys.stderr)

    if not args.inp or not args.outdir:
        ap.print_help()
        sys.exit(2)

    agg_path = pathlib.Path(args.inp)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_agg(str(agg_path))

    slope_path = pathlib.Path(args.slope) if args.slope else _default_slope_path(agg_path)
    slope = _maybe_read_slope(slope_path)

    pdf_path = outdir / "report.pdf"
    paths = []
    with PdfPages(pdf_path) as pdf:
        paths.append(plot_freq_ln_vs_A(df, outdir, args.fmt, args.dpi, slope, pdf))
        paths.append(plot_survival(df, outdir, args.fmt, args.dpi, pdf))
        paths.append(plot_freq_vs_A(df, outdir, args.fmt, args.dpi, pdf))
        paths.append(plot_cdf(df, outdir, args.fmt, args.dpi, pdf))

    html_path = outdir / "report.html"
    html_lines = ["<html><body>"]
    for p in paths:
        html_lines.append(f'<img src="{p.name}" alt="{p.name}"><br>')
    html_lines.append("</body></html>")
    html_path.write_text("\n".join(html_lines))

    bin_path = os.environ.get("ASSEMBLYMC_BIN", "AssemblyMC.exe")
    bin_name = os.path.basename(bin_path)
    trials = os.environ.get("TRIALS", "?")
    commit = os.environ.get("ASSEMBLYMC_COMMIT", "unknown")
    provenance = {
        "assemblymc": {
            "commit": commit,
            "cmd": f"{bin_name} --trials {trials}",
            "bin": bin_name,
        }
    }
    json_path = outdir / "report.json"
    json_path.write_text(json.dumps(provenance, indent=2))

    msg = [str(p) for p in paths]
    msg.extend([str(html_path), str(pdf_path), str(json_path)])
    print("wrote: " + "; ".join(msg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
