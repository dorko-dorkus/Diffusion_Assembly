
"""High-level experiment driver for the Diffusion Assembly project.

This script ties together the individual stages used throughout the experiments:

1. **AI generation** via :func:`assembly_diffusion.qm9_ai.generate_qm9_chon_ai`.
2. **Training** over all configuration variants located in ``configs``.
3. **Sampling** a single molecule using the trained policy.
4. **Analysis** using convenience utilities from :mod:`analysis`.

The stages are deliberately lightweight so that the script can be executed in
continuous integration environments.  A ``--smoke`` flag is provided to limit
training to a single epoch.
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import yaml
import torch

# Ensure the repository root is on the Python path so we can import the local
# modules when the script is executed as ``python scripts/experiment.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assembly_diffusion.qm9_ai import generate_qm9_chon_ai
from assembly_diffusion.config import load_config
from assembly_diffusion.data import get_dataloader
from assembly_diffusion.forward import ForwardKernel
from assembly_diffusion.mask import FeasibilityMask
from assembly_diffusion.backbone import GNNBackbone
from assembly_diffusion.policy import ReversePolicy
from assembly_diffusion.train import train_epoch
from analysis import ks_test, sensitivity_over_lambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual experiment stages
# ---------------------------------------------------------------------------

def run_ai_generation() -> None:
    """Generate assembly index annotations for the QM9 CHON subset."""
    logger.info("[AI generation] Creating QM9 annotations …")
    generate_qm9_chon_ai()


def train_all_configs(config_dir: str = "configs", *, smoke: bool = False) -> None:
    """Train models for all configuration variants found in ``config_dir``.

    Parameters
    ----------
    config_dir:
        Directory containing YAML configuration files.
    smoke:
        If ``True``, restricts training to a single epoch for fast execution.
    """

    cfg_path = Path(config_dir)
    for path in sorted(cfg_path.glob("*.yaml")):
        with open(path, "r", encoding="utf8") as f:
            raw = yaml.safe_load(f) or {}
        variants = [k for k in raw.keys() if k != "common"]
        for variant in variants:
            cfg = load_config(path, variant)
            logger.info("[Training] %s:%s", path.name, variant)

            loader = get_dataloader(batch_size=cfg.batch, max_heavy=cfg.max_atoms)
            kernel = ForwardKernel()
            mask = FeasibilityMask()
            policy = ReversePolicy(GNNBackbone(node_dim=cfg.hidden_dim))
            opt_cls = {
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
            }.get(cfg.optimiser.lower(), torch.optim.AdamW)
            optimiser = opt_cls(policy.parameters(), lr=cfg.lr)

            epochs = 1 if smoke else cfg.epochs
            for epoch in range(epochs):
                metrics = train_epoch(
                    loader,
                    kernel,
                    policy,
                    mask,
                    optimiser,
                    lambda_reg=cfg.guid_coeff,
                    epoch=epoch,
                )
                logger.info(
                    "  Epoch %d/%d - loss: %.3f acc: %.3f",
                    epoch + 1,
                    epochs,
                    metrics["loss"],
                    metrics["accuracy"],
                )


def run_sampling() -> None:
    """Draw a single sample from the current policy."""
    from assembly_diffusion.graph import MoleculeGraph
    from assembly_diffusion.sampler import Sampler

    logger.info("[Sampling] Drawing one molecule …")
    torch.manual_seed(0)
    x_init = MoleculeGraph(["C", "O"], torch.zeros((2, 2), dtype=torch.int64))
    kernel = ForwardKernel()
    mask = FeasibilityMask()
    policy = ReversePolicy(GNNBackbone())
    sampler = Sampler(policy, mask)
    x = sampler.sample(kernel.T, x_init, gamma=1.0)
    try:
        logger.info("  Sampled SMILES: %s", x.canonical_smiles())
    except Exception:
        logger.info("  Sampled molecule has no valid SMILES representation")


def run_analysis() -> None:
    """Run lightweight analysis on the generated data."""
    import pandas as pd

    logger.info("[Analysis] Running statistical checks …")
    path = ROOT / "qm9_chon_ai.csv"
    if not path.exists():
        logger.info("  No data available for analysis")
        return

    df = pd.read_csv(path)
    if df.empty:
        logger.info("  Dataset is empty – skipping analysis")
        return

    sample_a = df["ai_exact"].head(100)
    sample_b = df["ai_surrogate"].head(100)
    ks = ks_test(sample_a, sample_b)
    sens = sensitivity_over_lambda(df.head(5))
    logger.info(
        "  KS statistic: %.3f, p-value: %.3g",
        ks["statistic"],
        ks["pvalue"],
    )
    logger.info("  Sensitivity medians: %s", sens)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(smoke: bool = False) -> None:
    """Execute the full experiment pipeline."""
    run_ai_generation()
    train_all_configs(smoke=smoke)
    run_sampling()
    run_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a lightweight configuration for quick checks",
    )
    args = parser.parse_args()
    main(smoke=args.smoke)
