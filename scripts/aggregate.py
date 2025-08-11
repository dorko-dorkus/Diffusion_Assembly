"""Aggregate results placeholder script.

This thin wrapper will eventually collate outputs from earlier pipeline steps.
The current implementation only supports a ``--dry-run`` flag for interface
testing.
"""

from __future__ import annotations

import argparse
import logging


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate pipeline outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended actions without executing",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        logger.info("Dry run: no action taken")
        return 0

    logger.info("Aggregation step is not yet implemented")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
