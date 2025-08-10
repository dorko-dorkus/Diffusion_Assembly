"""Experiment runner for the assembly diffusion pipeline.

baseline: the default configuration and its reference metrics act as
    the baseline for subsequent comparisons.
data_sources: experiment settings and model paths are loaded from
    ``configs/registry.yaml`` along with generated artifacts.
metrics: pipeline evaluation statistics are written via
    ``metrics_writer`` such as validity and AI score summaries.
objective: execute a single experiment and persist a manifest with
    environment details for downstream analysis.
validation: smoke tests like ``tests/test_pipeline_smoke.py`` ensure the
    script runs end to end and produces metrics.
"""

import hashlib, json, os, subprocess, sys, time, yaml, logging
from datetime import datetime
from pathlib import Path

# ensure repository root on path for package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from assembly_diffusion.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _git_hash():
    """Return the current git commit hash or ``"unknown"``.

    When the project is executed from a zip archive the ``.git`` directory is
    absent and invoking ``git`` would emit an error message.  To keep logs clean
    we first check for ``.git`` and only then attempt to call ``git`` while
    silencing any stderr output.
    """

    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        return "unknown"

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=repo_root
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _pip_freeze():
    try:
        return (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            .decode()
            .splitlines()
        )
    except (subprocess.CalledProcessError, OSError):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    import argparse, random
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None

    from assembly_diffusion.eval.metrics_writer import write_metrics

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
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch is not None:
        torch.manual_seed(cfg["seed"])
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)

    # run the pipeline and always write metrics.json
    metrics, flags = run_pipeline(cfg, outdir)
    write_metrics(outdir, seed=int(cfg["seed"]), config=cfg, **metrics)

    _manifest(outdir, cfg, extra={"run_id": run_id, "requires_confirmation": flags})
    logger.info("[OK] Wrote manifest and artifacts to %s", outdir)
