import hashlib, json, os, subprocess, sys, time, yaml
from datetime import datetime
from pathlib import Path

# ensure repository root on path for package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _pip_freeze():
    try:
        return (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            .decode()
            .splitlines()
        )
    except Exception:
        return []


def _manifest(outdir, cfg, extra):
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


def run_pipeline(cfg, outdir):
    """
    Temporary stub to unblock smoke tests.
    Replace with the real sampler and evaluation, but always return a dict
    that matches the metrics writer signature.
    """
    # If you already generate samples.smi here, you can compute real metrics.
    # For now return neutral values.
    return {
        "valid_fraction": 0.0,
        "uniqueness": 0.0,
        "diversity": 0.0,
        "novelty": 0.0,
        "median_ai": 0.0,
    }


if __name__ == "__main__":
    import argparse, random
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    from assembly_diffusion.eval.metrics_writer import write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Experiment name key in configs/registry.yaml")
    p.add_argument("--outdir", default="results", help="Base results dir")
    args = p.parse_args()

    registry = yaml.safe_load(open("configs/registry.yaml"))
    assert args.name in registry["experiments"], f"Unknown experiment {args.name}"
    cfg = registry["experiments"][args.name]

    run_id = f"{args.name}_{int(time.time())}"
    outdir = os.path.join(args.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)

    # set seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch is not None:
        torch.manual_seed(cfg["seed"])

    # run the pipeline and always write metrics.json
    metrics = run_pipeline(cfg, outdir)
    write_metrics(outdir, **metrics)

    _manifest(outdir, cfg, extra={"run_id": run_id})
    print(f"[OK] Wrote manifest and artifacts to {outdir}")
