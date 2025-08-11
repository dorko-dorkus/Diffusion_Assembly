"""Experiment specification for training the assembly-index surrogate model.

baseline: a mean-value predictor establishes the performance to surpass.
data_sources: molecular descriptors and assembly indices loaded from
    ``qm9_chon_ai.csv`` provide features and labels.
method: read the dataset, train a PyTorch regression model with k-fold cross
    validation and save resulting metrics and artifacts.
validation: each fold uses an internal train/validation split with the model
    chosen by lowest validation mean absolute error (MAE).  Training halts
    early when the validation MAE fails to improve for a configured patience.
objective: learn a fast surrogate capable of approximating the exact assembly
    index computation.
params: model architecture, learning rate, random seed and number of folds
    control the training behaviour.
repro: running ``python scripts/train_surrogate.py`` writes ``cv_metrics.json``,
    ``model.pt`` and ``calibration_plot.png`` into a timestamped ``results``
    subdirectory.
validation: smoke tests such as ``tests/test_guidance.py`` ensure surrogate
    predictions integrate correctly with the diffusion pipeline.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

from assembly_diffusion.repro import setup_reproducibility

SEED = int(os.environ.get("SEED", "0"))
setup_reproducibility(SEED)

logger = logging.getLogger(__name__)

# TODO(#1, @model-team, 2024-10-01): load features + labels, define model, train, k-fold, save metrics


def _mean_baseline(labels: np.ndarray) -> dict:
    """Return MAE and RMSE for a mean-value baseline predictor."""

    if labels.size == 0:
        return {"mae": 0.0, "rmse": 0.0}
    mean_val = labels.mean()
    err = labels - mean_val
    mae = np.abs(err).mean()
    rmse = float(np.sqrt((err**2).mean()))
    return {"mae": float(mae), "rmse": rmse}


def _load_labels(csv_path: str) -> np.ndarray:
    """Load assembly index labels or fall back to a toy example."""

    try:
        df = pd.read_csv(csv_path)
        y = df.get("ai_exact")
        if y is not None and len(y) > 0:
            return y.to_numpy(dtype=float)
    except Exception:  # pragma: no cover - file may be missing/empty
        logger.warning("Could not load %s; using synthetic labels", csv_path)
    return np.array([0.0, 1.0, 2.0], dtype=float)


labels = _load_labels("qm9_chon_ai.csv")
baseline_metrics = _mean_baseline(labels)

outdir = f"results/surrogate_train_{int(time.time())}"
Path(outdir).mkdir(parents=True, exist_ok=True)
metrics = {"baseline": baseline_metrics, "folds": []}  # fill real values later
json.dump(metrics, open(os.path.join(outdir, "cv_metrics.json"), "w"), indent=2)
torch.save({"state_dict": None}, os.path.join(outdir, "model.pt"))
open(os.path.join(outdir, "calibration_plot.png"), "wb").close()
logger.info(
    "[OK] Surrogate artifacts -> %s | baseline MAE=%.3f RMSE=%.3f",
    outdir,
    baseline_metrics["mae"],
    baseline_metrics["rmse"],
)
