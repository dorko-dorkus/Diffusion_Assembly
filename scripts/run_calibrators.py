from __future__ import annotations
import argparse
import pandas as pd
from assembly_diffusion.calibrators.sampler import Sampler
from assembly_diffusion.monitor import CSVLogger

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
        df = S.sample_S(L_max=args.L_max, n_samp=args.n_samp, guided=args.guided, gamma=args.gamma)
    else:
        df = S.sample_T(N_max=args.N_max, n_samp=args.n_samp, guided=args.guided, gamma=args.gamma, dmin_exact=args.dmin_exact)
    freq = df.groupby(["id","universe","grammar","As_lower","As_upper","validity","d_min"]).size().reset_index(name="count")
    freq["frequency"] = freq["count"] / freq["count"].sum()
    freq = freq.drop(columns=["count"])
    freq.to_csv(args.out, index=False)
    print(f"Wrote {len(freq)} rows to {args.out}")
