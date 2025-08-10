"""Experiment specification for the Assembly Diffusion CLI.

baseline: unguided diffusion model from the ``exp01_baseline`` registry entry.
data_sources: QM9-CHON subset provided in :mod:`qm9_chon_ai.csv`.
method: invoke :func:`assembly_diffusion.cli.main` to parse registry
    configuration and run the experiment pipeline.
metrics: each run writes ``metrics.json`` with validity, assembly index and
    related scores.
objective: demonstrate launching and evaluating a configured experiment.
params: experiment parameters come from ``configs/registry.yaml`` and any CLI
    overrides.
repro: deterministic seeds and committed configs ensure reproducibility.
validation: ``scripts/check_registry.py`` and unit tests verify experiment
    setup.
"""

from .cli import main

if __name__ == "__main__":
    main()
