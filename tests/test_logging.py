import json
from pathlib import Path

from assembly_diffusion.logging import setup_run_logger


def test_run_log_contains_json_header(tmp_path):
    log_file = tmp_path / "run.log"
    seeds = {"python": 0, "numpy": 1}
    config = {"lr": 0.1}
    logger = setup_run_logger(log_file, grammar="G", config=config, seeds=seeds)
    logger.info("start")
    lines = log_file.read_text().splitlines()
    assert len(lines) >= 1
    header_line = lines[0]
    header = json.loads(header_line)
    # header must contain all required keys
    for key in ("seeds", "package_versions", "commit", "grammar", "config"):
        assert key in header
    assert header["seeds"] == seeds
    assert header["grammar"] == "G"
    assert header["config"] == config
    # header should appear in first five lines
    assert header_line in lines[:5]
