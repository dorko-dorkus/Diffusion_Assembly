"""Configuration loading utilities for training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, get_type_hints

import warnings
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


@dataclass
class AIConfig:
    """Configuration options for assembly index estimation."""

    method: str = "surrogate"
    trials: int = 1
    timeout_s: float = 0.0


@dataclass
class RunConfig:
    """Top level run configuration."""

    seeds: List[int]
    N_samp: int
    guidance: float
    ai: AIConfig


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


def load_run_config(path: str | Path) -> RunConfig:
    """Load a YAML run configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file describing a run.  Expected keys are
        ``seeds`` (list of ints), ``N_samp`` (int), ``guidance`` (float) and
        an ``ai`` section with ``method``, ``trials`` and ``timeout_s``.
    """

    with open(Path(path), "r", encoding="utf8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    seeds_raw = raw.get("seeds", [])
    seeds = [int(s) for s in (seeds_raw if isinstance(seeds_raw, list) else [seeds_raw])]

    ai_raw = raw.get("ai", {})
    method = str(ai_raw.get("method", "surrogate"))
    if method == "exact":
        warnings.warn(
            "ai.method='exact' is deprecated; use 'assemblymc' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "assemblymc"
    ai_cfg = AIConfig(
        method=method,
        trials=int(ai_raw.get("trials", 1)),
        timeout_s=float(ai_raw.get("timeout_s", 0.0)),
    )

    cfg = RunConfig(
        seeds=seeds,
        N_samp=int(raw.get("N_samp", 0)),
        guidance=float(raw.get("guidance", 0.0)),
        ai=ai_cfg,
    )
    return cfg
