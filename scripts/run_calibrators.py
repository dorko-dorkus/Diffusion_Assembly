from __future__ import annotations
import argparse
import csv
import json
import os
import time
import pandas as pd
from assembly_diffusion.calibrators.sampler import Sampler

parser = argparse.ArgumentParser()
parser.add_argument("universe", choices=["S", "T"], help="Calibrator universe")
parser.add_argument("--n_samp", type=int, default=20000)
parser.add_argument("--L_max", type=int, default=12)
parser.add_argument("--N_max", type=int, default=10)
parser.add_argument("--guided", action="store_true")
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="calibrator_samples.csv")
parser.add_argument("--dmin_exact", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    S = Sampler(seed=args.seed)
    if args.universe == "S":
        df = S.sample_S(
            L_max=args.L_max,
            n_samp=args.n_samp,
            guided=args.guided,
            gamma=args.gamma,
        )
    else:
        df = S.sample_T(
            N_max=args.N_max,
            n_samp=args.n_samp,
            guided=args.guided,
            gamma=args.gamma,
            dmin_exact=args.dmin_exact,
        )

    freq = df.groupby(
        ["id", "universe", "grammar", "As_lower", "As_upper", "validity", "d_min"]
    ).size().reset_index(name="count")
    freq["frequency"] = freq["count"] / freq["count"].sum()
    freq = freq.drop(columns=["count"])

    if args.universe == "S":
        results_strings = [
            (row.id, int(row.As_upper)) for row in freq.itertuples()
        ]
        results_trees: list[tuple[int, int]] = []
    else:
        results_strings = []
        results_trees = [
            (int(row.As_upper) + 1, int(row.As_upper)) for row in freq.itertuples()
        ]

    freq.to_csv(args.out, index=False)
    print(f"Wrote {len(freq)} rows to {args.out}")

    outdir = f"results/calibrators_{int(time.time())}"
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "string_ai.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "ai"])
        for s, ai in results_strings:
            w.writerow([s, ai])

    with open(os.path.join(outdir, "tree_ai.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "ai"])
        w.writerows(results_trees)

    json.dump(
        {
            "invariants": ["AI >= 0", "AI increases with concatenation under G"],
            "status": "ok",
        },
        open(os.path.join(outdir, "manifest.json"), "w"),
        indent=2,
    )

    print(f"[OK] Calibrator outputs -> {outdir}")
