import json
from pathlib import Path

from assembly_diffusion.run_logger import init_run_logger, reset_run_logger
from assembly_diffusion.eval.reporting import write_guided_unguided_report


def test_write_guided_unguided_report(tmp_path):
    log_path = tmp_path / "run.log"
    init_run_logger(str(log_path), "G", {})
    guided = {"m": 1.2, "CI": [1.0, 1.4], "validity": 0.9, "S(A)": 0.1}
    unguided = {"m": 0.8, "CI": [0.5, 1.0], "validity": 0.85, "S(A)": 0.2}
    json_p, md_p = write_guided_unguided_report(str(tmp_path), guided, unguided)

    data = json.loads(Path(json_p).read_text())
    assert data["guided"]["m"] == 1.2
    assert data["unguided"]["validity"] == 0.85

    md = Path(md_p).read_text()
    assert "| m | 1.2 | 0.8 |" in md
    assert "| S(A) | 0.1 | 0.2 |" in md

    log_text = log_path.read_text()
    assert "guided_vs_unguided.json" in log_text
    assert "guided_vs_unguided.md" in log_text
    reset_run_logger()
