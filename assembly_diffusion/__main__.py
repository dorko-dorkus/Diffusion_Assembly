"""Experiment specification for the Assembly Diffusion CLI.

baseline: unguided diffusion model from the ``exp01_baseline`` registry entry.
data_sources: QM9-CHON subset provided in :mod:`qm9_chon_ai.csv`.
method: invoke :func:`assembly_diffusion.cli.main` to parse registry
    configuration and run the experiment pipeline.
metrics: each run writes ``metrics.json`` with validity, assembly index and
    related scores. Classification quality uses the F1 score
    ``F1 = 2 * TP / (2 * TP + FP + FN)`` while regression error is measured
    with the mean squared error ``MSE = \frac{1}{N} \sum (y_i - \hat{y}_i)^2``.
    Results across multiple random seeds are reported as ``mean ± std``.
objective: demonstrate launching and evaluating a configured experiment.
params: experiment parameters come from ``configs/registry.yaml`` and any CLI
    overrides.
repro: deterministic seeds and committed configs ensure reproducibility.
validation: experiments rely on an 80/10/10 train/validation/test split with
    the best checkpoint chosen by validation F1.  Training may stop early when
    that score fails to improve for a set patience.  ``scripts/check_registry.py``
    and unit tests verify experiment setup.
"""

import argparse

from .cli import main as cli_main
from .repro import setup_reproducibility


def run_baseline() -> None:
    """Execute a simple unguided sampling run as a control baseline."""

    cli_main(["sample"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Assembly Diffusion wrapper")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run an additional unguided baseline sample before executing the requested command.",
    )
    args, rest = parser.parse_known_args()
    setup_reproducibility(0)
    if args.baseline:
        run_baseline()
    cli_main(rest)


if __name__ == "__main__":
    main()
