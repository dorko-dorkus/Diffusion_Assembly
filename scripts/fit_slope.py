"""Fit slope analysis placeholder.

This command line entry point will eventually fit slopes to aggregated results.
For now it only exposes a ``--dry-run`` option.
"""

from __future__ import annotations

import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit slopes to results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing the fitting procedure",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        logger.info("Dry run: no action taken")
        return 0

    logger.info("Slope fitting is not yet implemented")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
