"""Side-by-side reporting for guided vs unguided metrics.

baseline: unguided diffusion metrics provide the reference for comparison
    against guided runs.
data_sources: metric dictionaries containing slope ``m`` with confidence
    interval, validity fraction, and survival function ``S(A)`` values.
method: write a combined JSON and Markdown report comparing guided and
    unguided metrics side-by-side. The function logs the report paths using
    ``assembly_diffusion.run_logger`` so that run logs reference the outputs.
objective: simplify downstream analysis by storing a machine-readable JSON
    summary and a human-friendly Markdown table in the same directory.
params: ``outdir`` destination directory, ``guided`` and ``unguided`` metric
    dictionaries.
repro: output is deterministic given the metric inputs.
validation: :mod:`tests.test_reporting` covers JSON/Markdown contents and log
    references.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, Tuple


def _fmt_ci(ci: Any) -> str:
    """Return a string representation of a confidence interval."""
    if not ci:
        return "N/A"
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    return str(ci)


def write_guided_unguided_report(
    outdir: str, guided: Dict[str, Any], unguided: Dict[str, Any]
) -> Tuple[str, str]:
    """Write side-by-side JSON and Markdown reports.

    Parameters
    ----------
    outdir:
        Directory in which the reports will be written. Created if necessary.
    guided, unguided:
        Metric dictionaries for the guided and unguided runs respectively.

    Returns
    -------
    tuple of str
        Paths to the JSON and Markdown files.
    """

    payload = {"guided": guided, "unguided": unguided}
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "guided_vs_unguided.json")
    md_path = os.path.join(outdir, "guided_vs_unguided.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = [
        ("m", guided.get("m"), unguided.get("m")),
        ("CI", _fmt_ci(guided.get("CI")), _fmt_ci(unguided.get("CI"))),
        ("validity", guided.get("validity"), unguided.get("validity")),
        ("S(A)", guided.get("S(A)"), unguided.get("S(A)")),
    ]
    md_lines = ["| Metric | Guided | Unguided |", "|---|---|---|"]
    for label, gv, uv in rows:
        md_lines.append(f"| {label} | {gv} | {uv} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger = logging.getLogger("assembly_diffusion.run")
    logger.info("wrote guided_vs_unguided report to %s", json_path)
    logger.info("wrote guided_vs_unguided report to %s", md_path)

    return json_path, md_path


__all__ = ["write_guided_unguided_report"]
