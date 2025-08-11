"""Compute AI scores placeholder.

This script is a stub for computing assembly indices.  The ``--dry-run`` option
allows the command line interface to be exercised without executing any core
logic.
"""

from __future__ import annotations

import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute assembly indices")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without performing any computation",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        logger.info("Dry run: no action taken")
        return 0

    logger.info("AI computation is not yet implemented")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
