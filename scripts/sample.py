"""Sample pipeline step wrapper.

This thin script is a placeholder for the sampling stage of the diffusion
assembly pipeline.  It intentionally avoids importing heavy dependencies so
that the ``--dry-run`` flag can be used for quick CLI verification.
"""

from __future__ import annotations

import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the sample step.

    Parameters
    ----------
    argv:
        Optional list of command line arguments.  When ``None`` the arguments
        are pulled from ``sys.argv``.
    """

    parser = argparse.ArgumentParser(description="Run the sampling step")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        logger.info("Dry run: no action taken")
        return 0

    logger.info("Sampling step is not yet implemented")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
