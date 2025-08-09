import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from assembly_diffusion.eval.metrics_writer import write_metrics


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        return "unknown"


def _pip_freeze() -> list[str]:
    try:
        return (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])\
            .decode()
            .splitlines()
        )
    except Exception:
        return []


def _manifest(outdir: str | os.PathLike, cfg: dict, extra: dict) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_hash": _git_hash(),
        "python": sys.version,
        "requirements": _pip_freeze(),
        "config": cfg,
        "extra": extra,
    }
    with open(Path(outdir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--name", required=True, help="Experiment name key in configs/registry.yaml"
    )
    p.add_argument("--outdir", default="results", help="Base results dir")
    args = p.parse_args()

    registry = yaml.safe_load(open("configs/registry.yaml"))
    assert args.name in registry["experiments"], f"Unknown experiment {args.name}"
    cfg = registry["experiments"][args.name]

    run_id = f"{args.name}_{int(time.time())}"
    outdir = os.path.join(args.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)

    # set seeds
    import random
    import numpy as np
    import torch

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # TODO: integrate your existing training/sampling here. Pseudocode:
    # samples, metrics, ai_scores = run_pipeline(cfg)

    # Placeholder structure for writer calls you already have:
    # save SMILES
    # with open(os.path.join(outdir, "samples.smi"), "w") as f: ...
    # save metrics
    # write_metrics(
    #     outdir,
    #     valid_fraction=...,  # fraction of valid structures
    #     uniqueness=...,  # fraction of unique structures
    #     diversity=...,  # pairwise Tanimoto diversity
    #     novelty=...,  # fraction not seen in training
    #     median_ai=...,  # median AI score
    # )
    # save AI scores
    # np.savetxt(os.path.join(outdir, "ai_scores.csv"), ai_scores, delimiter=",")

    _manifest(outdir, cfg, extra={"run_id": run_id})
    print(f"[OK] Wrote manifest and artifacts to {outdir}")

