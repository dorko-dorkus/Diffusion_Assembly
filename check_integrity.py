import os
import subprocess
import sys
import glob
from pathlib import Path

import torch


def main() -> None:
    """Verify that checkpoints were created from the current commit.

    For every ``*.pt`` file in the ``checkpoints`` directory, this script
    expects a ``commit`` entry containing the git hash of the source code used
    to create the checkpoint.  The hash must match the repository's HEAD.
    """
    repo_root = Path(__file__).resolve().parent
    if not (repo_root / ".git").exists():
        print("Unable to determine git commit: missing .git directory", file=sys.stderr)
        raise SystemExit(1)

    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=repo_root
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as exc:
        print(f"Unable to determine git commit: {exc}", file=sys.stderr)
        raise SystemExit(1)

    ok = True
    for path in glob.glob(os.path.join("checkpoints", "*.pt")):
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - best effort load
            print(f"Failed to load {path}: {exc}", file=sys.stderr)
            ok = False
            continue
        commit = data.get("commit")
        if commit != head:
            print(
                f"Commit mismatch for {path}: expected {head}, got {commit}",
                file=sys.stderr,
            )
            ok = False
    if not ok:
        raise SystemExit(1)

    print("Checkpoint integrity verified")


if __name__ == "__main__":
    main()
