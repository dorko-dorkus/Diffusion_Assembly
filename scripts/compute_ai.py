#!/usr/bin/env python3
"""
Compute assembly indices for input SMILES and write a protocol-ready CSV.

Usage (matches your runner):
  python scripts/compute_ai.py --in <samples.csv|parquet> --out <ai.csv> \
    --method {assemblymc|surrogate} --trials 2000 --timeout-s 2 [--seed 0] [--bin-path <path>] [--allow-fallback]

Input schema (from sample.py): id,smiles
Output schema (consumed downstream): id,smiles,As_lower,As_upper,d_min,validity,method,status,elapsed_ms
"""
from __future__ import annotations
import argparse, os, sys, time, pathlib, traceback
import pandas as pd

# Optional RDKit validity check
try:
    from rdkit import Chem
except Exception:
    Chem = None

# Optional AssemblyMC adapter (only used if --method assemblymc)
def _load_assemblymc():
    try:
        from assembly_diffusion.extern.assemblymc import a_star_and_dmin  # type: ignore
        return a_star_and_dmin
    except Exception as e:
        return None

def _read_any(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"--in not found: {p}")
    ext = p.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p)
    sys.exit(f"Unsupported input extension: {ext}")

def _write_csv(df: pd.DataFrame, out_path: str):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def _valid_smiles(smiles: str) -> bool:
    if Chem is None:
        # If RDKit unavailable, assume valid to avoid hard-fail in CI; downstream can re-check.
        return True
    m = Chem.MolFromSmiles(smiles, sanitize=True)
    return m is not None

def _surrogate_A(smiles: str) -> int:
    """Very simple heuristic fallback so pipeline can run without AssemblyMC."""
    if Chem is None:
        # Count tokens as a crude proxy
        return max(1, sum(c.isalpha() for c in smiles) // 2)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    # Heuristic: atoms minus 1 (lower-bias), clamp >=0
    return max(0, m.GetNumAtoms() - 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV/Parquet with id,smiles")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--method", choices=["assemblymc", "surrogate"], required=True)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--timeout-s", type=int, default=2, dest="timeout_s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-path", type=str, default=None,
                    help="Optional path to AssemblyMC binary (env ASSEMBLYMC_BIN takes precedence)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="If assemblymc unavailable, fall back to surrogate instead of exiting")
    args = ap.parse_args()

    df = _read_any(args.inp)
    if not {"id", "smiles"}.issubset(df.columns):
        sys.exit("Input must contain columns: id,smiles")

    rows = []
    use_mc = args.method == "assemblymc"
    a_star_and_dmin = None
    mc_bin = None

    if use_mc:
        # Resolve binary path (permission model)
        mc_bin = os.environ.get("ASSEMBLYMC_BIN") or args.bin_path
        if not mc_bin or not pathlib.Path(mc_bin).exists():
            msg = ("AssemblyMC binary not found. Set ASSEMBLYMC_BIN to your locally built AssemblyMC.exe "
                   "or use --bin-path. (We do not redistribute binaries.)")
            if args.allow_fallback:
                print("WARN:", msg, "Falling back to --method surrogate.", file=sys.stderr)
                use_mc = False
            else:
                sys.exit(msg)
        else:
            os.environ["ASSEMBLYMC_BIN"] = mc_bin  # ensure adapter sees it
            a_star_and_dmin = _load_assemblymc()
            if a_star_and_dmin is None:
                if args.allow_fallback:
                    print("WARN: AssemblyMC adapter not importable; falling back to surrogate.", file=sys.stderr)
                    use_mc = False
                else:
                    sys.exit("AssemblyMC adapter not importable (assembly_diffusion.extern.assemblymc).")

    total = len(df)
    ok = 0
    start_all = time.perf_counter()

    for i, r in df.iterrows():
        sid = str(r["id"])
        smi = str(r["smiles"])
        t0 = time.perf_counter()
        status = "ok"
        validity = 1 if _valid_smiles(smi) else 0
        A_lo = A_hi = 0
        d_min = None

        try:
            if use_mc and validity:
                res = a_star_and_dmin(smi, trials=args.trials, seed=args.seed, timeout_s=args.timeout_s)  # type: ignore
                A_star, d_min_est, stats = res
                A_lo = int(A_star)
                A_hi = int(A_star)
                d_min = int(d_min_est) if d_min_est is not None else None
            else:
                A_est = _surrogate_A(smi)
                A_lo = int(max(0, A_est - 1))
                A_hi = int(A_est + 1)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = "error"
            # Keep going; record failure for this row
            print(f"ERR [{sid}]: {e}", file=sys.stderr)
            # traceback.print_exc()

        elapsed_ms = int(1000 * (time.perf_counter() - t0))
        rows.append({
            "id": sid,
            "smiles": smi,
            "As_lower": A_lo,
            "As_upper": A_hi,
            "d_min": d_min,
            "validity": int(validity),
            "method": "assemblymc" if use_mc and validity and status == "ok" else "surrogate",
            "status": status,
            "elapsed_ms": elapsed_ms,
        })
        if status == "ok":
            ok += 1

    out_df = pd.DataFrame(rows)
    _write_csv(out_df, args.out)

    dur_s = time.perf_counter() - start_all
    print(f"processed={total} ok={ok} wrote={args.out} time_s={dur_s:.2f}")

if __name__ == "__main__":
    main()
