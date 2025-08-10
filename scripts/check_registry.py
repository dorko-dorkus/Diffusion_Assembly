#!/usr/bin/env python3
"""Validate experiment registry for label-to-scorer consistency."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def main() -> None:
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

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    print("Registry OK")


if __name__ == "__main__":
    main()
