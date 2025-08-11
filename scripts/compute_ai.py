#!/usr/bin/env python3
"""
Compute assembly indices for input SMILES and write a protocol-ready CSV.

Accepted flags (all optional synonyms supported):
  --in / --input / -i              Path to samples (CSV/Parquet) with columns: id,smiles
  --out / --output / -o            Output CSV path
  --method {assemblymc|surrogate}
  --trials INT
  --timeout-s INT  (alias: --timeout_s)
  --seed INT
  --bin-path PATH  (alias: --bin_path)
  --allow-fallback  (or env ALLOW_FALLBACK=1)

Unknown extra args are ignored (logged to stderr), so runner changes don’t break execution.
"""
from __future__ import annotations
import argparse, os, sys, time, pathlib, json
import pandas as pd


# ---- Helpers ----------------------------------------------------------------
def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=True)
    # Primary names
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--input", dest="inp")
    ap.add_argument("--output", dest="out")
    ap.add_argument("-i", dest="inp")
    ap.add_argument("-o", dest="out")
    ap.add_argument("--method", choices=["assemblymc", "surrogate"], required=False, default="assemblymc")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=2)
    ap.add_argument("--timeout_s", dest="timeout_s", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-path", dest="bin_path")
    ap.add_argument("--bin_path", dest="bin_path")
    ap.add_argument("--allow-fallback", dest="allow_fallback", action="store_true")
    return ap


def _read_any(p: str) -> pd.DataFrame:
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


def _write_csv(df: pd.DataFrame, out_path: str):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


# Optional RDKit validity check (soft dependency)
try:
    from rdkit import Chem
except Exception:
    Chem = None


def _valid_smiles(s: str) -> bool:
    if Chem is None:
        return True
    return Chem.MolFromSmiles(s, sanitize=True) is not None


def _surrogate_A(s: str) -> int:
    if Chem is None:
        return max(1, sum(c.isalpha() for c in s) // 2)
    m = Chem.MolFromSmiles(s)
    return 0 if m is None else max(0, m.GetNumAtoms() - 1)


def _load_mc():
    try:
        from assembly_diffusion.extern.assemblymc import a_star_and_dmin  # type: ignore
        return a_star_and_dmin
    except Exception:
        return None


# ---- Main -------------------------------------------------------------------
def main() -> int:
    ap = _arg_parser()
    args, unknown = ap.parse_known_args()
    if unknown:
        print("WARN compute_ai.py: ignoring extra args:", " ".join(unknown), file=sys.stderr)

    # Required IO
    if not args.inp or not args.out:
        ap.print_help()
        sys.exit(2)

    df = _read_any(args.inp)
    if not {"id", "smiles"}.issubset(df.columns):
        sys.exit("Input must contain columns: id,smiles")

    # Resolve method & permission model
    method = args.method or "assemblymc"
    allow_fallback = args.allow_fallback or (
        os.environ.get("ALLOW_FALLBACK", "") not in ("", "0", "false", "False")
    )
    use_mc = method == "assemblymc"
    mc_func = None

    if use_mc:
        mc_bin = os.environ.get("ASSEMBLYMC_BIN") or args.bin_path
        if not (mc_bin and pathlib.Path(mc_bin).exists()):
            msg = (
                "AssemblyMC binary not found. Set ASSEMBLYMC_BIN to your locally built AssemblyMC.exe "
                "or use --bin-path. (We do not redistribute binaries.)"
            )
            if allow_fallback:
                print("WARN:", msg, "Falling back to surrogate.", file=sys.stderr)
                use_mc = False
            else:
                sys.exit(msg)
        else:
            os.environ["ASSEMBLYMC_BIN"] = mc_bin
            mc_func = _load_mc()
            if mc_func is None:
                if allow_fallback:
                    print(
                        "WARN: AssemblyMC adapter not importable; falling back to surrogate.",
                        file=sys.stderr,
                    )
                    use_mc = False
                else:
                    sys.exit(
                        "AssemblyMC adapter not importable (assembly_diffusion.extern.assemblymc)."
                    )

    rows = []
    t_all = time.perf_counter()
    ok = 0
    for _, r in df.iterrows():
        sid = str(r["id"])
        smi = str(r["smiles"])
        t0 = time.perf_counter()
        status = "ok"
        validity = 1 if _valid_smiles(smi) else 0
        A_lo = A_hi = 0
        d_min = None
        try:
            if use_mc and validity:
                A_star, d_min_est, stats = mc_func(  # type: ignore
                    smi, trials=args.trials, seed=args.seed, timeout_s=args.timeout_s
                )
                A_lo = int(A_star)
                A_hi = int(A_star)
                d_min = int(d_min_est) if d_min_est is not None else None
            else:
                A_est = _surrogate_A(smi)
                A_lo, A_hi = max(0, A_est - 1), A_est + 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = "error"
            print(f"ERR [{sid}]: {e}", file=sys.stderr)

        elapsed_ms = int(1000 * (time.perf_counter() - t0))
        rows.append(
            {
                "id": sid,
                "smiles": smi,
                "As_lower": int(A_lo),
                "As_upper": int(A_hi),
                "d_min": d_min,
                "validity": int(validity),
                "method": "assemblymc"
                if (use_mc and validity and status == "ok")
                else "surrogate",
                "status": status,
                "elapsed_ms": elapsed_ms,
            }
        )
        if status == "ok":
            ok += 1

    out_df = pd.DataFrame(rows)
    _write_csv(out_df, args.out)
    print(
        json.dumps(
            {
                "processed": len(df),
                "ok": ok,
                "out": args.out,
                "time_s": round(time.perf_counter() - t_all, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

