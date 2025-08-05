"""Configuration loading utilities for training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainConfig:
    dataset: str
    max_atoms: int
    hidden_dim: int
    optimiser: str
    lr: float
    batch: int
    epochs: int
    schedule: str
    guid_coeff: float
    guid_mode: str


def load_config(path: str | Path, variant: str) -> TrainConfig:
    """Load a YAML configuration file and return merged settings.

    Parameters
    ----------
    path:
        Path to a YAML file with ``common`` settings and model variants.
    variant:
        Key selecting one of the variant sections in the YAML file.
    """

    with open(Path(path), "r", encoding="utf8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    common = raw.get("common", {})
    if variant not in raw:
        raise KeyError(f"Unknown variant '{variant}' in configuration")
    specific = raw.get(variant, {})
    merged: Dict[str, Any] = {**common, **specific}

    missing = [k for k in TrainConfig.__annotations__ if k not in merged]
    if missing:
        raise KeyError(f"Missing keys in configuration: {missing}")

    return TrainConfig(**merged)
