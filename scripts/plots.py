"""Plot generation placeholder script.

The plotting step will visualise results from the pipeline.  At present the
script merely provides a ``--dry-run`` flag for interface testing.
"""

from __future__ import annotations

import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without creating any plots",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        logger.info("Dry run: no action taken")
        return 0

    logger.info("Plot generation is not yet implemented")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
