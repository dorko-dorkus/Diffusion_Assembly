#!/usr/bin/env python3
"""Validate experiment registry for label-to-scorer consistency."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    registry_path = Path(__file__).resolve().parent.parent / "configs" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    experiments = data.get("experiments", {})

    errors: list[str] = []
    for name, cfg in experiments.items():
        desc = cfg.get("description", "")
        scorer = cfg.get("ai", {}).get("scorer")
        if re.search(r"\bexact\b", desc, re.IGNORECASE) and scorer != "exact":
            errors.append(
                f"{name}: description indicates exact AI but ai.scorer={scorer!r}"
            )
        if re.search(r"\bsurrogate\b", desc, re.IGNORECASE) and scorer != "surrogate":
            errors.append(
                f"{name}: description indicates surrogate AI but ai.scorer={scorer!r}"
            )

    if errors:
        for err in errors:
            logger.error(err)
        raise SystemExit(1)

    logger.info("Registry OK")


if __name__ == "__main__":
    main()
