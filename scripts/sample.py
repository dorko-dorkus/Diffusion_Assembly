#!/usr/bin/env python3
"""
Sample N SMILES and write to --out as CSV or Parquet.

Priority for sources:
1) --source <file> with a 'smiles' column (csv/parquet/tsv).
2) qm9_chon_ai.csv in repo root (if present, uses its 'smiles' column).
3) Built-in small fallback set (simple organics) for smoke tests.

Output schema: id,smiles
"""
import argparse, os, sys, random, pathlib
import pandas as pd

FALLBACK_SMILES = [
    "C", "CC", "CCC", "CCCC", "C=C", "C#C", "c1ccccc1", "c1ccncc1",
    "CCO", "CCN", "CCCl", "CCBr", "CC=O", "O=C(O)C", "CC(C)O", "CC(C)N",
    "C1CCCCC1", "C1=CC=CC=C1", "CCOC", "CCCN", "CCSC", "COC", "CCOCC",
    "CC(C)C", "N#CC", "CC(=O)O", "OCCO", "c1ccco1", "c1ccsc1", "c1ncccc1",
    "CC#N", "CC(=O)N"
]

def load_source(path: str | None) -> pd.Series:
    # 1) explicit --source
    if path:
        p = pathlib.Path(path)
        if not p.exists():
            sys.exit(f"--source not found: {p}")
        df = read_any(p)
        return pick_smiles(df)
    # 2) repo default file
    for candidate in ["qm9_chon_ai.csv", "data/qm9_chon_ai.csv"]:
        p = pathlib.Path(candidate)
        if p.exists():
            df = pd.read_csv(p)
            return pick_smiles(df)
    # 3) fallback
    return pd.Series(FALLBACK_SMILES, name="smiles")

def read_any(p: pathlib.Path) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(p)
    if ext in [".tsv"]:
        return pd.read_csv(p, sep="\t")
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    sys.exit(f"Unsupported source extension: {ext}")

def pick_smiles(df: pd.DataFrame) -> pd.Series:
    for col in ["smiles", "SMILES", "Smiles"]:
        if col in df.columns:
            s = df[col].dropna().astype(str)
            s = s[s.str.len() > 0]
            if len(s) == 0:
                break
            return s
    sys.exit("Source file missing a 'smiles' column with data.")

def write_any(df: pd.DataFrame, out_path: str):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext in [".csv", ".txt"]:
        df.to_csv(out, index=False)
        return
    if ext in [".parquet", ".pq"]:
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            sys.stderr.write(
                f"Parquet write failed ({e}). "
                "Install 'pyarrow' or output to .csv instead.\n"
            )
            sys.exit(2)
        return
    sys.exit(f"Unsupported output extension: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="Number of samples to emit")
    ap.add_argument("--out", type=str, required=True, help="Output path (.csv or .parquet)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--source", type=str, default=None, help="Optional source file with 'smiles' column")
    args = ap.parse_args()

    random.seed(args.seed)
    src = load_source(args.source).tolist()
    if len(src) == 0:
        sys.exit("No SMILES available from source.")

    # Sample with replacement; assign deterministic ids
    picks = [random.choice(src) for _ in range(args.n)]
    df = pd.DataFrame({
        "id": [f"mol_{i:06d}" for i in range(args.n)],
        "smiles": picks,
    })

    write_any(df, args.out)

    # Minimal stdout summary for logs
    print(f"wrote {len(df)} rows to {args.out}")
    print(f"unique_smiles={df['smiles'].nunique()}")

if __name__ == "__main__":
    main()
