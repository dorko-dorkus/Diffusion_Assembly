"""Configuration loading utilities for training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, get_type_hints

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


@dataclass
class SamplingConfig:
    """Configuration options for sampling with guidance."""

    guidance_gamma: float = 0.0
    guidance_mode: str = "A_lower"
    max_steps: int = 64
    delta_valid_tol: float = 0.05


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

    # ``yaml.safe_load`` returns plain scalars as strings in some cases
    # (e.g. ``3e-4``).  Cast the merged configuration values to the types
    # expected by :class:`TrainConfig` to avoid type errors downstream.
    hints = get_type_hints(TrainConfig)
    for key, typ in hints.items():
        if key in merged:
            try:
                merged[key] = typ(merged[key])
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Invalid value for '{key}': {merged[key]}") from exc

    return TrainConfig(**merged)
