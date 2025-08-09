import os
import subprocess
import sys
import glob
from pathlib import Path
import logging

import torch

logger = logging.getLogger(__name__)


def main() -> None:
    """Verify that checkpoints were created from the current commit.

    For every ``*.pt`` file in the ``checkpoints`` directory, this script
    expects a ``commit`` entry containing the git hash of the source code used
    to create the checkpoint.  The hash must match the repository's HEAD.
    """
    repo_root = Path(__file__).resolve().parent
    if not (repo_root / ".git").exists():
        logger.error("Unable to determine git commit: missing .git directory")
        raise SystemExit(1)

    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=repo_root
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.error("Unable to determine git commit: %s", exc)
        raise SystemExit(1)

    ok = True
    for path in glob.glob(os.path.join("checkpoints", "*.pt")):
        try:
            data = torch.load(path, map_location="cpu")
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort load
            logger.error("Failed to load %s: %s", path, exc)
            ok = False
            continue
        commit = data.get("commit")
        if commit != head:
            logger.error(
                "Commit mismatch for %s: expected %s, got %s", path, head, commit
            )
            ok = False
    if not ok:
        raise SystemExit(1)

    logger.info("Checkpoint integrity verified")


if __name__ == "__main__":
    main()
